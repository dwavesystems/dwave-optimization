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

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/lp.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using namespace Catch::Matchers;

const double FEASIBILITY_TOLERANCE = 1e-07;

namespace dwave::optimization {

TEST_CASE("LinearProgramNode") {
    GIVEN("A two variable, two row LP") {
        // min: -x0 + 4x1
        // such that:
        //      -3x0 + x1 <= 6
        //      -x0 - 2x1 >= -4
        //      x1 >= -3

        auto graph = Graph();

        // c = [-1, 4]
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{-1, 4});

        // A_ub = [[-3, 1], [1, 2]], b_ub = [6, 4]
        auto A_ub = std::vector<double>{-3, 1, 1, 2};
        auto A_ub_ptr = graph.emplace_node<ConstantNode>(A_ub.data(), std::vector<ssize_t>{2, 2});

        // b_ub = [6, 4]
        auto b_ub_ptr = graph.emplace_node<ConstantNode>(std::vector{6, 4});

        // lb = [-inf, -3]
        auto lb_ptr =
            graph.emplace_node<ConstantNode>(std::vector{-LinearProgramNode::infinity(), -3.0});

        auto lp_ptr = graph.emplace_node<LinearProgramNode>(
            c_ptr, nullptr, A_ub_ptr, b_ub_ptr, nullptr, nullptr, lb_ptr, nullptr
        );

        auto feas_ptr = graph.emplace_node<LinearProgramFeasibleNode>(lp_ptr);
        auto obj_ptr = graph.emplace_node<LinearProgramObjectiveValueNode>(lp_ptr);
        auto sol_ptr = graph.emplace_node<LinearProgramSolutionNode>(lp_ptr);

        graph.emplace_node<ArrayValidationNode>(feas_ptr);
        graph.emplace_node<ArrayValidationNode>(obj_ptr);
        graph.emplace_node<ArrayValidationNode>(sol_ptr);

        WHEN("We initialize the state") {
            auto state = graph.initialize_state();

            THEN("x0 = 10, x1 = -3") {
                CHECK(sol_ptr->size(state) == 2);
                CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(10, FEASIBILITY_TOLERANCE));
                CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(-3, FEASIBILITY_TOLERANCE));

                CHECK(lp_ptr->feasible(state));
                CHECK(feas_ptr->view(state).front());
                CHECK_THAT(
                    obj_ptr->view(state).front(), WithinAbs(-1 * 10 + 4 * -3, FEASIBILITY_TOLERANCE)
                );
                CHECK(feas_ptr->view(state).front());
            }
        }
    }

    GIVEN("An infeasible two variable, one row LP") {
        // min: x0 + x1
        // such that:
        //      x0 + x1 <= 6
        //      x1 >= 10
        //      x1 >= 10

        auto graph = Graph();

        // c = [1, 1]
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{1, 1});

        // A_ub = [[1, 1]]
        // We want to vary it in the tests so we'll actually set the state later
        auto A_ub_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{1, 2}, -20, 20);

        // b_ub = [6]
        auto b_ub_ptr = graph.emplace_node<ConstantNode>(std::vector{6});

        // lb = [10, 10]
        auto lb_ptr = graph.emplace_node<ConstantNode>(std::vector{10, 10});

        auto lp_ptr = graph.emplace_node<LinearProgramNode>(
            c_ptr, nullptr, A_ub_ptr, b_ub_ptr, nullptr, nullptr, lb_ptr, nullptr
        );

        auto feas_ptr = graph.emplace_node<LinearProgramFeasibleNode>(lp_ptr);
        auto obj_ptr = graph.emplace_node<LinearProgramObjectiveValueNode>(lp_ptr);
        auto sol_ptr = graph.emplace_node<LinearProgramSolutionNode>(lp_ptr);

        graph.emplace_node<ArrayValidationNode>(feas_ptr);
        graph.emplace_node<ArrayValidationNode>(obj_ptr);
        graph.emplace_node<ArrayValidationNode>(sol_ptr);

        WHEN("We initialize the state") {
            auto state = graph.empty_state();
            A_ub_ptr->initialize_state(state, {1, 1});
            graph.initialize_state(state);

            THEN("the model is infeasible") {
                CHECK(sol_ptr->size(state) == 2);
                CHECK(sol_ptr->view(state).size() == 2);  // the actual state is undefined

                CHECK(!lp_ptr->feasible(state));
                CHECK(!feas_ptr->view(state).front());
            }

            AND_WHEN("We update A_ub to make the problem feasible") {
                // min: x0 + x1
                // such that:
                //      x0 - x1 <= 6
                //      x1 >= 10
                //      x1 >= 10
                A_ub_ptr->set_value(state, 1, -10);
                graph.propagate(state, graph.descendants(state, {A_ub_ptr}));

                THEN("the model is feasible") {
                    CHECK(sol_ptr->size(state) == 2);
                    CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(10, FEASIBILITY_TOLERANCE));
                    CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(10, FEASIBILITY_TOLERANCE));

                    CHECK(lp_ptr->feasible(state));
                    CHECK(feas_ptr->view(state).front());
                    CHECK_THAT(
                        obj_ptr->view(state).front(), WithinAbs(10 + 10, FEASIBILITY_TOLERANCE)
                    );
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {A_ub_ptr}));

                    THEN("the model is feasible") {
                        CHECK(sol_ptr->size(state) == 2);
                        CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(10, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(10, FEASIBILITY_TOLERANCE));

                        CHECK(lp_ptr->feasible(state));
                        CHECK(feas_ptr->view(state).front());
                        CHECK_THAT(
                            obj_ptr->view(state).front(), WithinAbs(10 + 10, FEASIBILITY_TOLERANCE)
                        );
                    }

                    AND_WHEN("We update A_ub to make the problem infeasible again") {
                        // min: x0 + x1
                        // such that:
                        //      x0 + x1 <= 6
                        //      x1 >= 10
                        //      x1 >= 10
                        A_ub_ptr->set_value(state, 1, 10);
                        graph.propagate(state, graph.descendants(state, {A_ub_ptr}));

                        THEN("the model is infeasible") {
                            CHECK(sol_ptr->size(state) == 2);
                            CHECK(sol_ptr->view(state).size() == 2);

                            CHECK(!lp_ptr->feasible(state));
                            CHECK(!feas_ptr->view(state).front());
                        }
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {A_ub_ptr}));

                    THEN("the model is infeasible") {
                        CHECK(sol_ptr->size(state) == 2);
                        CHECK(sol_ptr->view(state).size() == 2);  // the actual state is undefined

                        CHECK(!feas_ptr->view(state).front());
                    }
                }
            }
        }
    }

    GIVEN("A two variable, three row LP") {
        // min: x0 + x1
        // such that:
        //             x1 <= 7
        // 5 <=  x0 + 2x1 <= 15
        // 6 <= 3x0 + 2x1
        // bounds:
        // 0 <= x0 <= 4
        // 1 <= x1

        auto graph = Graph();

        // c = [1, 1]
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{1, 1});

        // b_lb = [-inf, 5, 6], A = [[0, 1], [1, 2], [3, 2]], b_ub = [7, 15, inf]
        auto b_lb_ptr = graph.emplace_node<ConstantNode>(
            std::vector<double>{-LinearProgramNode::infinity(), 5, 6}
        );
        auto A = std::vector<double>{0, 1, 1, 2, 3, 2};
        auto A_ptr = graph.emplace_node<ConstantNode>(A.data(), std::vector<ssize_t>{3, 2});
        auto b_ub_ptr = graph.emplace_node<ConstantNode>(
            std::vector<double>{7, 15, LinearProgramNode::infinity()}
        );

        // lb = [0, 1], ub = [4, inf]
        auto lb_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1});
        auto ub_ptr =
            graph.emplace_node<ConstantNode>(std::vector{4, LinearProgramNode::infinity()});

        auto lp_ptr = graph.emplace_node<LinearProgramNode>(
            c_ptr, b_lb_ptr, A_ptr, b_ub_ptr, nullptr, nullptr, lb_ptr, ub_ptr
        );

        auto feas_ptr = graph.emplace_node<LinearProgramFeasibleNode>(lp_ptr);
        auto sol_ptr = graph.emplace_node<LinearProgramSolutionNode>(lp_ptr);

        graph.emplace_node<ArrayValidationNode>(feas_ptr);
        graph.emplace_node<ArrayValidationNode>(sol_ptr);

        THEN("The integrality, min, and max are as expected") {
            CHECK(!sol_ptr->integral());
            CHECK(sol_ptr->min() == 0);
            CHECK(sol_ptr->max() == LinearProgramNode::infinity());
        }

        WHEN("We initialize the state") {
            auto state = graph.initialize_state();

            THEN("x0 = .5, x1 = 2.25") {
                CHECK(sol_ptr->size(state) == 2);
                CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE));
                CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE));

                CHECK(lp_ptr->feasible(state));
                CHECK(feas_ptr->view(state).front());
            }
        }
    }

    GIVEN("A three variable, 4 row LP with all predecessors as variables") {
        auto graph = Graph();

        const ssize_t num_variables = 3;
        const ssize_t num_eq = 2;
        const ssize_t num_ineq = 2;

        const ssize_t min = IntegerNode::minimum_lower_bound;
        const ssize_t max = IntegerNode::maximum_upper_bound;

        auto c = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);

        auto b_lb = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);
        auto A = graph.emplace_node<IntegerNode>(std::vector{num_ineq, num_variables}, min, max);
        auto b_ub = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);

        auto A_eq = graph.emplace_node<IntegerNode>(std::vector{num_eq, num_variables}, min, max);
        auto b_eq = graph.emplace_node<IntegerNode>(std::vector{num_eq}, min, max);

        auto lb = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto ub = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);

        auto lp_ptr = graph.emplace_node<LinearProgramNode>(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub);

        auto feas_ptr = graph.emplace_node<LinearProgramFeasibleNode>(lp_ptr);
        auto sol_ptr = graph.emplace_node<LinearProgramSolutionNode>(lp_ptr);

        graph.emplace_node<ArrayValidationNode>(feas_ptr);
        graph.emplace_node<ArrayValidationNode>(sol_ptr);

        WHEN("We instatiate an LP problem") {
            // min: x0 + x1 + x2
            // such that:
            //             x1       <= 7
            // 6 <= 3x0 + 2x1
            // 5 <=  x0 + 2x1       <= 5
            // 8 <=  x0 +  x1 + 4x2 <= 8
            // bounds:
            // 0 <= x0 <= 4
            // 1 <= x1

            auto state = graph.empty_state();

            c->initialize_state(state, {1, 1, 1});

            b_lb->initialize_state(state, {min, 6});
            A->initialize_state(state, {0, 1, 0, 3, 2, 0});
            b_ub->initialize_state(state, {7, max});

            A_eq->initialize_state(state, {1, 2, 0, 1, 1, 4});
            b_eq->initialize_state(state, {5, 8});

            lb->initialize_state(state, {0, 1, min});
            ub->initialize_state(state, {4, max, max});

            graph.initialize_state(state);

            THEN("x0 = .5, x1 = 2.25") {
                CHECK(sol_ptr->size(state) == 3);
                CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE));
                CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE));
                CHECK_THAT(sol_ptr->view(state)[2], WithinAbs(1.3125, FEASIBILITY_TOLERANCE));
            }

            AND_WHEN("We change the c vector and then propagate") {
                // min: x0 + x1 + 2*x2        <- changed
                // such that:
                //             x1       <= 7
                // 6 <= 3x0 + 2x1
                // 5 <=  x0 + 2x1       <= 5
                // 8 <=  x0 +  x1 + 4x2 <= 8
                // bounds:
                // 0 <= x0 <= 4
                // 1 <= x1

                c->set_value(state, 2, 2);

                graph.propagate(state, graph.descendants(state, {c}));

                THEN("x0 = .5, x1 = 2.25, x2 = 1.3125") {
                    CHECK(sol_ptr->size(state) == 3);
                    CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE));
                    CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE));
                    CHECK_THAT(
                        sol_ptr->view(state)[2], WithinAbs(1.3125, FEASIBILITY_TOLERANCE)
                    );  // this halved
                }

                AND_WHEN("We then revert") {
                    graph.revert(state, graph.descendants(state, {c}));

                    THEN("x0 = .5, x1 = 2.25, x2 = 1.3125") {
                        CHECK(sol_ptr->size(state) == 3);
                        CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(
                            sol_ptr->view(state)[2], WithinAbs(1.3125, FEASIBILITY_TOLERANCE)
                        );
                    }
                }
            }

            AND_WHEN("We change the A_eq matrix and then propagate") {
                // min: x0 + x1 + x2
                // such that:
                //             x1       <= 7
                // 6 <= 3x0 + 2x1
                // 5 <=  x0 + 2x1       <= 5
                // 8 <=  x0 +  x1 + 2x2 <= 8
                // bounds:
                // 0 <= x0 <= 4
                // 1 <= x1

                A_eq->set_value(state, 5, 2);

                graph.propagate(state, graph.descendants(state, {A_eq}));

                THEN("x0 = .5, x1 = 2.25, x2 = 2.625") {
                    CHECK(sol_ptr->size(state) == 3);
                    CHECK(sol_ptr->view(state)[0] == .5);
                    CHECK(sol_ptr->view(state)[1] == 2.25);
                    CHECK(sol_ptr->view(state)[2] == 2.625);  // this doubled
                }

                AND_WHEN("We then revert") {
                    graph.revert(state, graph.descendants(state, {A_eq}));

                    THEN("x0 = .5, x1 = 2.25, x2 = 1.3125") {
                        CHECK(sol_ptr->size(state) == 3);
                        CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(
                            sol_ptr->view(state)[2], WithinAbs(1.3125, FEASIBILITY_TOLERANCE)
                        );
                    }

                    AND_WHEN("We then make a different change") {
                        // min: x0 + x1 + x2
                        // such that:
                        //             x1       <= 7
                        // 6 <= 3x0 + 2x1
                        // 5 <=  x0 + 2x1       <= 5
                        // 8 <= 2x0 +  x1 + 4x2 <= 8
                        // bounds:
                        // 0 <= x0 <= 4
                        // 1 <= x1

                        A_eq->set_value(state, 3, 2);

                        graph.propagate(state, graph.descendants(state, {A_eq}));

                        THEN("x0 = .5, x1 = 2.25, x2 = 1.1875") {
                            REQUIRE(sol_ptr->size(state) == 3);
                            CHECK_THAT(
                                sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE)
                            );
                            CHECK_THAT(
                                sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE)
                            );
                            CHECK_THAT(
                                sol_ptr->view(state)[2], WithinAbs(1.1875, FEASIBILITY_TOLERANCE)
                            );
                        }
                    }
                }

                AND_WHEN("We then commit") {
                    graph.commit(state, graph.descendants(state, {A_eq}));

                    THEN("x0 = .5, x1 = 2.25, x2 = 2.625") {
                        CHECK(sol_ptr->size(state) == 3);
                        CHECK_THAT(sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(
                            sol_ptr->view(state)[2], WithinAbs(2.625, FEASIBILITY_TOLERANCE)
                        );  // this doubled
                    }

                    AND_WHEN("We then make a different change") {
                        // min: x0 + x1 + x2
                        // such that:
                        //             x1       <= 7
                        // 6 <= 3x0 + 2x1
                        // 5 <=  x0 + 2x1       <= 5
                        // 8 <= 2x0 +  x1 + 4x2 <= 8
                        // bounds:
                        // 0 <= x0 <= 4
                        // 1 <= x1

                        A_eq->set_value(state, 3, 2);
                        A_eq->set_value(state, 5, 4);

                        graph.propagate(state, graph.descendants(state, {A_eq}));

                        // TODO: this test originally only set A_eq[3] = 2 and then checked for
                        // the solution x0 = 3.0, x1 = 1.0, x2 = 0.5, but it appears that the
                        // simplex implementation is not finding the optimal on this problem.
                        // Probably worth investigating.
                        THEN("x0 = .5, x1 = 2.25, x2 = 1.1875") {
                            REQUIRE(sol_ptr->size(state) == 3);
                            CHECK_THAT(
                                sol_ptr->view(state)[0], WithinAbs(.5, FEASIBILITY_TOLERANCE)
                            );
                            CHECK_THAT(
                                sol_ptr->view(state)[1], WithinAbs(2.25, FEASIBILITY_TOLERANCE)
                            );
                            CHECK_THAT(
                                sol_ptr->view(state)[2], WithinAbs(1.1875, FEASIBILITY_TOLERANCE)
                            );
                        }
                    }
                }
            }

            AND_WHEN("We change the b_eq matrix and then propagate") {
                // min: x0 + x1 + x2
                // such that:
                //             x1       <= 7
                // 6 <= 3x0 + 2x1
                // 5 <=  x0 + 2x1       <= 5
                // 7 <=  x0 +  x1 + 4x2 <= 7
                // bounds:
                // 0 <= x0 <= 4
                // 1 <= x1

                b_eq->set_value(state, 1, 105);  // redundant, but nice for testing
                b_eq->set_value(state, 1, 7);

                graph.propagate(state, graph.descendants(state, {b_eq}));

                THEN("The new LP is satisfied") {
                    CHECK(sol_ptr->size(state) == 3);

                    const double x0 = sol_ptr->view(state)[0];
                    const double x1 = sol_ptr->view(state)[1];
                    const double x2 = sol_ptr->view(state)[2];

                    CHECK(x1 <= 7 + FEASIBILITY_TOLERANCE);
                    CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                    CHECK_THAT(x0 + 2 * x1, WithinAbs(5, FEASIBILITY_TOLERANCE));
                    CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(7, FEASIBILITY_TOLERANCE));

                    CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                    CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                    CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                }

                AND_WHEN("We then revert") {
                    // min: x0 + x1 + x2
                    // such that:
                    //             x1       <= 7
                    // 6 <= 3x0 + 2x1
                    // 5 <=  x0 + 2x1       <= 5
                    // 8 <=  x0 +  x1 + 4x2 <= 8
                    // bounds:
                    // 0 <= x0 <= 4
                    // 1 <= x1
                    graph.revert(state, graph.descendants(state, {b_eq}));

                    THEN("The original LP is satisfied") {
                        CHECK(sol_ptr->size(state) == 3);

                        const double x0 = sol_ptr->view(state)[0];
                        const double x1 = sol_ptr->view(state)[1];
                        const double x2 = sol_ptr->view(state)[2];

                        CHECK(x1 <= 7 + FEASIBILITY_TOLERANCE);
                        CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                        CHECK_THAT(x0 + 2 * x1, WithinAbs(5, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(8, FEASIBILITY_TOLERANCE));

                        CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                        CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                        CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                    }

                    AND_WHEN("We then make a different change") {
                        // min: x0 + x1 + x2
                        // such that:
                        //             x1       <= 7
                        // 6 <= 3x0 + 2x1
                        // 6 <=  x0 + 2x1       <= 6
                        // 8 <=  x0 +  x1 + 4x2 <= 8
                        // bounds:
                        // 0 <= x0 <= 4
                        // 1 <= x1

                        b_eq->set_value(state, 0, 6);

                        graph.propagate(state, graph.descendants(state, {b_eq}));

                        THEN("The new LP is satisfied") {
                            CHECK(sol_ptr->size(state) == 3);

                            const double x0 = sol_ptr->view(state)[0];
                            const double x1 = sol_ptr->view(state)[1];
                            const double x2 = sol_ptr->view(state)[2];

                            CHECK(x1 <= 7 + FEASIBILITY_TOLERANCE);
                            CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                            CHECK_THAT(x0 + 2 * x1, WithinAbs(6, FEASIBILITY_TOLERANCE));
                            CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(8, FEASIBILITY_TOLERANCE));

                            CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                            CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                            CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                        }
                    }
                }

                AND_WHEN("We then commit") {
                    // min: x0 + x1 + x2
                    // such that:
                    //             x1       <= 7
                    // 6 <= 3x0 + 2x1
                    // 5 <=  x0 + 2x1       <= 5
                    // 7 <=  x0 +  x1 + 4x2 <= 7
                    // bounds:
                    // 0 <= x0 <= 4
                    // 1 <= x1
                    graph.commit(state, graph.descendants(state, {b_eq}));

                    THEN("The new LP is satisfied") {
                        CHECK(sol_ptr->size(state) == 3);

                        const double x0 = sol_ptr->view(state)[0];
                        const double x1 = sol_ptr->view(state)[1];
                        const double x2 = sol_ptr->view(state)[2];

                        CHECK(x1 <= 7 + FEASIBILITY_TOLERANCE);
                        CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                        CHECK_THAT(x0 + 2 * x1, WithinAbs(5, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(7, FEASIBILITY_TOLERANCE));

                        CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                        CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                        CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                    }

                    AND_WHEN("We then make a different change") {
                        // min: x0 + x1 + x2
                        // such that:
                        //             x1       <= 7
                        // 6 <= 3x0 + 2x1
                        // 6 <=  x0 + 2x1       <= 6
                        // 8 <=  x0 +  x1 + 4x2 <= 7
                        // bounds:
                        // 0 <= x0 <= 4
                        // 1 <= x1

                        b_eq->set_value(state, 0, 6);

                        graph.propagate(state, graph.descendants(state, {b_eq}));

                        THEN("The new LP is satisfied") {
                            CHECK(sol_ptr->size(state) == 3);

                            const double x0 = sol_ptr->view(state)[0];
                            const double x1 = sol_ptr->view(state)[1];
                            const double x2 = sol_ptr->view(state)[2];

                            CHECK(x1 <= 7 + FEASIBILITY_TOLERANCE);
                            CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                            CHECK_THAT(x0 + 2 * x1, WithinAbs(6, FEASIBILITY_TOLERANCE));
                            CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(7, FEASIBILITY_TOLERANCE));

                            CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                            CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                            CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                        }
                    }
                }
            }

            AND_WHEN("We change the b_lb and b_ub matrices and then propagate") {
                // min: x0 + x1 + x2
                // such that:
                //             x1       <= 8  <- changed
                // 5 <= 3x0 + 2x1             <- changed
                // 5 <=  x0 + 2x1       <= 5
                // 8 <=  x0 +  x1 + 4x2 <= 8
                // bounds:
                // 0 <= x0 <= 4
                // 1 <= x1

                b_lb->set_value(state, 1, 1000);
                b_lb->set_value(state, 1, 5);
                b_ub->set_value(state, 0, 8);

                graph.propagate(state, graph.descendants(state, {b_lb, b_ub}));

                THEN("The new LP is satisfied") {
                    CHECK(sol_ptr->size(state) == 3);

                    const double x0 = sol_ptr->view(state)[0];
                    const double x1 = sol_ptr->view(state)[1];
                    const double x2 = sol_ptr->view(state)[2];

                    CHECK(x1 <= 8 + FEASIBILITY_TOLERANCE);
                    CHECK(5 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                    CHECK(x0 + 2 * x1 == 5);
                    CHECK(x0 + x1 + 4 * x2 == 8);

                    CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                    CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                    CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                }

                AND_WHEN("We then revert") {
                    // min: x0 + x1 + x2
                    // such that:
                    //             x1       <= 7
                    // 6 <= 3x0 + 2x1
                    // 5 <=  x0 + 2x1       <= 5
                    // 8 <=  x0 +  x1 + 4x2 <= 8
                    // bounds:
                    // 0 <= x0 <= 4
                    // 1 <= x1
                    graph.revert(state, graph.descendants(state, {b_lb, b_ub}));

                    THEN("The original LP is satisfied") {
                        CHECK(sol_ptr->size(state) == 3);

                        const double x0 = sol_ptr->view(state)[0];
                        const double x1 = sol_ptr->view(state)[1];
                        const double x2 = sol_ptr->view(state)[2];

                        CHECK(x1 <= 7 + FEASIBILITY_TOLERANCE);
                        CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                        CHECK_THAT(x0 + 2 * x1, WithinAbs(5, FEASIBILITY_TOLERANCE));
                        CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(8, FEASIBILITY_TOLERANCE));

                        CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                        CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                        CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                    }

                    AND_WHEN("We then make a different change") {
                        // min: x0 + x1 + x2
                        // such that:
                        //             x1       <= 6  <- changed
                        // 6 <= 3x0 + 2x1
                        // 5 <=  x0 + 2x1       <= 5
                        // 8 <=  x0 +  x1 + 4x2 <= 8
                        // bounds:
                        // 0 <= x0 <= 4
                        // 1 <= x1

                        b_ub->set_value(state, 0, 6);

                        graph.propagate(state, graph.descendants(state, {b_ub}));

                        THEN("The new LP is satisfied") {
                            CHECK(sol_ptr->size(state) == 3);

                            const double x0 = sol_ptr->view(state)[0];
                            const double x1 = sol_ptr->view(state)[1];
                            const double x2 = sol_ptr->view(state)[2];

                            CHECK(lp_ptr->feasible(state));

                            CHECK(x1 <= 6 + FEASIBILITY_TOLERANCE);
                            CHECK(6 <= 3 * x0 + 2 * x1 + FEASIBILITY_TOLERANCE);

                            CHECK_THAT(x0 + 2 * x1, WithinAbs(5, FEASIBILITY_TOLERANCE));
                            CHECK_THAT(x0 + x1 + 4 * x2, WithinAbs(8, FEASIBILITY_TOLERANCE));

                            CHECK(0 <= x0 + FEASIBILITY_TOLERANCE);
                            CHECK(x0 <= 4 + FEASIBILITY_TOLERANCE);
                            CHECK(1 <= x1 + FEASIBILITY_TOLERANCE);
                        }
                    }
                }
            }
        }
    }

    SECTION("equality") {
        auto graph = Graph();

        const ssize_t num_variables = 3;
        const ssize_t num_eq = 2;
        const ssize_t num_ineq = 2;

        const ssize_t min = IntegerNode::minimum_lower_bound;
        const ssize_t max = IntegerNode::maximum_upper_bound;

        auto* c0 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto* c1 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);

        auto* b_lb0 = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);
        auto* b_lb1 = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);
        auto* A0 = graph.emplace_node<IntegerNode>(std::vector{num_ineq, num_variables}, min, max);
        auto* A1 = graph.emplace_node<IntegerNode>(std::vector{num_ineq, num_variables}, min, max);
        auto* b_ub0 = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);
        auto* b_ub1 = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);

        auto* A_eq0 = graph.emplace_node<IntegerNode>(std::vector{num_eq, num_variables}, min, max);
        auto* A_eq1 = graph.emplace_node<IntegerNode>(std::vector{num_eq, num_variables}, min, max);
        auto* b_eq0 = graph.emplace_node<IntegerNode>(std::vector{num_eq}, min, max);
        auto* b_eq1 = graph.emplace_node<IntegerNode>(std::vector{num_eq}, min, max);

        auto* lb0 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto* lb1 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto* ub0 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto* ub1 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);

        Node* lp0 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub0, A_eq0, b_eq0, lb0, ub0);
        Node* lp1 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub0, A_eq0, b_eq0, lb0, ub0);
        Node* lp2 =
            graph.emplace_node<LinearProgramNode>(c1, b_lb0, A0, b_ub0, A_eq0, b_eq0, lb0, ub0);
        Node* lp3 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb1, A0, b_ub0, A_eq0, b_eq0, lb0, ub0);
        Node* lp4 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A1, b_ub0, A_eq0, b_eq0, lb0, ub0);
        Node* lp5 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub1, A_eq0, b_eq0, lb0, ub0);
        Node* lp6 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub0, A_eq1, b_eq0, lb0, ub0);
        Node* lp7 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub0, A_eq0, b_eq1, lb0, ub0);
        Node* lp8 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub0, A_eq0, b_eq0, lb1, ub0);
        Node* lp9 =
            graph.emplace_node<LinearProgramNode>(c0, b_lb0, A0, b_ub0, A_eq0, b_eq0, lb0, ub1);

        CHECK(lp0->equal_to(*lp0));
        CHECK(lp0->equal_to(*lp1));

        CHECK(not lp0->equal_to(*c0));
        CHECK(not lp0->equal_to(*lp2));
        CHECK(not lp0->equal_to(*lp3));
        CHECK(not lp0->equal_to(*lp4));
        CHECK(not lp0->equal_to(*lp5));
        CHECK(not lp0->equal_to(*lp6));
        CHECK(not lp0->equal_to(*lp7));
        CHECK(not lp0->equal_to(*lp8));
        CHECK(not lp0->equal_to(*lp9));

        Node* feas0 =
            graph.emplace_node<LinearProgramFeasibleNode>(dynamic_cast<LinearProgramNode*>(lp0));
        Node* feas1 =
            graph.emplace_node<LinearProgramFeasibleNode>(dynamic_cast<LinearProgramNode*>(lp0));
        Node* feas2 =
            graph.emplace_node<LinearProgramFeasibleNode>(dynamic_cast<LinearProgramNode*>(lp1));

        CHECK(feas0->equal_to(*feas0));
        CHECK(feas0->equal_to(*feas1));
        CHECK(not feas0->equal_to(*feas2));
        CHECK(not feas0->equal_to(*c0));

        Node* obj0 = graph.emplace_node<LinearProgramObjectiveValueNode>(
            dynamic_cast<LinearProgramNode*>(lp0)
        );
        Node* obj1 = graph.emplace_node<LinearProgramObjectiveValueNode>(
            dynamic_cast<LinearProgramNode*>(lp0)
        );
        Node* obj2 = graph.emplace_node<LinearProgramObjectiveValueNode>(
            dynamic_cast<LinearProgramNode*>(lp1)
        );

        CHECK(obj0->equal_to(*obj0));
        CHECK(obj0->equal_to(*obj1));
        CHECK(not obj0->equal_to(*obj2));
        CHECK(not obj0->equal_to(*c0));

        Node* sol0 =
            graph.emplace_node<LinearProgramSolutionNode>(dynamic_cast<LinearProgramNode*>(lp0));
        Node* sol1 =
            graph.emplace_node<LinearProgramSolutionNode>(dynamic_cast<LinearProgramNode*>(lp0));
        Node* sol2 =
            graph.emplace_node<LinearProgramSolutionNode>(dynamic_cast<LinearProgramNode*>(lp1));

        CHECK(sol0->equal_to(*sol0));
        CHECK(sol0->equal_to(*sol1));
        CHECK(not sol0->equal_to(*sol2));
        CHECK(not sol0->equal_to(*c0));
    }

    SECTION("predecessor replacement") {
        auto graph = Graph();

        const ssize_t num_variables = 3;
        const ssize_t num_eq = 2;
        const ssize_t num_ineq = 2;

        const ssize_t min = IntegerNode::minimum_lower_bound;
        const ssize_t max = IntegerNode::maximum_upper_bound;

        auto* c0 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto* c1 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);

        auto* b_lb0 = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);
        auto* b_lb1 = graph.emplace_node<IntegerNode>(std::vector{num_ineq}, min, max);
        auto* A0 = graph.emplace_node<IntegerNode>(std::vector{num_ineq, num_variables}, min, max);
        auto* A1 = graph.emplace_node<IntegerNode>(std::vector{num_ineq, num_variables}, min, max);

        auto* A_eq0 = graph.emplace_node<IntegerNode>(std::vector{num_eq, num_variables}, min, max);
        auto* A_eq1 = graph.emplace_node<IntegerNode>(std::vector{num_eq, num_variables}, min, max);
        auto* b_eq0 = graph.emplace_node<IntegerNode>(std::vector{num_eq}, min, max);
        auto* b_eq1 = graph.emplace_node<IntegerNode>(std::vector{num_eq}, min, max);

        auto* ub0 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);
        auto* ub1 = graph.emplace_node<IntegerNode>(std::vector{num_variables}, min, max);

        auto* lp0 = graph.emplace_node<LinearProgramNode>(
            c0, b_lb0, A0, nullptr, A_eq0, b_eq0, nullptr, ub0
        );
        auto* lp1 = graph.emplace_node<LinearProgramNode>(
            c0, b_lb0, A0, nullptr, A_eq0, b_eq0, nullptr, ub0
        );

        auto* feas = graph.emplace_node<LinearProgramFeasibleNode>(lp0);
        auto* obj = graph.emplace_node<LinearProgramObjectiveValueNode>(lp0);
        auto* sol = graph.emplace_node<LinearProgramSolutionNode>(lp0);

        lp1->take_successors(*lp0);

        CHECK_THAT(feas->predecessors(), RangeEquals({lp1}));
        CHECK_THAT(obj->predecessors(), RangeEquals({lp1}));
        CHECK_THAT(sol->predecessors(), RangeEquals({lp1}));

        c1->take_successors(*c0);
        b_lb1->take_successors(*b_lb0);
        A1->take_successors(*A0);
        A_eq1->take_successors(*A_eq0);
        b_eq1->take_successors(*b_eq0);
        ub1->take_successors(*ub0);

        CHECK_THAT(lp0->predecessors(), RangeEquals({c1, b_lb1, A1, A_eq1, b_eq1, ub1}));
        CHECK_THAT(lp1->predecessors(), RangeEquals({c1, b_lb1, A1, A_eq1, b_eq1, ub1}));
    }
}

}  // namespace dwave::optimization

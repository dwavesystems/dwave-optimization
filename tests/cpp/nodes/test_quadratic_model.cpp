// Copyright 2023 D-Wave Inc.
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

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/quadratic_model.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("QuadraticModel") {
    QuadraticModel qmodel(10);

    GIVEN("An empty quadratic model of non-zero size") {
        THEN("Number of variables is noted") { CHECK(qmodel.num_variables() == 10); }

        THEN("All biases are zero") {
            CHECK(qmodel.get_linear(5) == 0);
            CHECK(qmodel.get_quadratic(5, 5) == 0);
            CHECK(qmodel.get_quadratic(5, 6) == 0);
        }

        WHEN("We set some biases") {
            qmodel.set_linear(5, 31.5);
            qmodel.set_quadratic(5, 5, 51);
            qmodel.set_quadratic(0, 5, 61.0);
            qmodel.set_quadratic(2, 5, 61.2);
            qmodel.set_quadratic(1, 5, 61.1);

            THEN("They are retrieved properly") {
                CHECK(qmodel.get_linear(5) == 31.5);
                CHECK(qmodel.get_quadratic(5, 5) == 51);
                CHECK(qmodel.get_quadratic(0, 5) == 61.0);
                CHECK(qmodel.get_quadratic(5, 0) == 61.0);
                CHECK(qmodel.get_quadratic(2, 5) == 61.2);
                CHECK(qmodel.get_quadratic(5, 2) == 61.2);
                CHECK(qmodel.get_quadratic(1, 5) == 61.1);
                CHECK(qmodel.get_quadratic(5, 1) == 61.1);
            }

            WHEN("We overwrite the biases") {
                qmodel.set_linear(5, 30.5);
                qmodel.set_quadratic(5, 5, 50);
                qmodel.set_quadratic(0, 5, 60.0);
                qmodel.set_quadratic(2, 5, 60.2);
                qmodel.set_quadratic(1, 5, 60.1);

                THEN("They are overwritten properly") {
                    CHECK(qmodel.get_linear(5) == 30.5);
                    CHECK(qmodel.get_quadratic(5, 5) == 50.0);
                    CHECK(qmodel.get_quadratic(5, 0) == 60.0);
                    CHECK(qmodel.get_quadratic(2, 5) == 60.2);
                    CHECK(qmodel.get_quadratic(5, 2) == 60.2);
                    CHECK(qmodel.get_quadratic(1, 5) == 60.1);
                    CHECK(qmodel.get_quadratic(5, 1) == 60.1);
                }
            }

            WHEN("We update the biases") {
                qmodel.add_linear(5, 31.5);
                qmodel.add_quadratic(5, 5, 51.0);
                qmodel.add_quadratic(0, 5, 61.0);
                qmodel.add_quadratic(2, 5, 61.2);
                qmodel.add_quadratic(1, 5, 61.1);

                THEN("They are updated properly") {
                    CHECK(qmodel.get_linear(5) == 63);
                    CHECK(qmodel.get_quadratic(5, 5) == 102.0);
                    CHECK(qmodel.get_quadratic(0, 5) == 122.0);
                    CHECK(qmodel.get_quadratic(5, 0) == 122.0);
                    CHECK(qmodel.get_quadratic(2, 5) == 122.4);
                    CHECK(qmodel.get_quadratic(5, 2) == 122.4);
                    CHECK(qmodel.get_quadratic(1, 5) == 122.2);
                    CHECK(qmodel.get_quadratic(5, 1) == 122.2);
                }
            }
        }
    }
}

TEST_CASE("QuadraticModelNode") {
    auto graph = Graph();
    QuadraticModel qmodel(5);

    // Linear =     { 0, 1, 2, 3, 4 }
    //
    // Quadratic =  { 0, 1, 2, 3, 4 }
    //              { 1, 2, 3, 4, 5 }
    //              { 2, 3, 4, 5, 6 }
    //              { 3, 4, 5, 6, 7 }
    //              { 4, 5, 6, 7, 8 }

    for (auto i = 0; i < 5; i++) {
        qmodel.add_linear(i, i);
        for (auto j = i; j < 5; j++) {
            qmodel.add_quadratic(i, j, i + j);
        }
    }

    GIVEN("A QuadraticModelNode with 5 binary variables") {
        auto binary_node_ptr = graph.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{5});
        auto qnode_ptr = graph.emplace_node<QuadraticModelNode>(binary_node_ptr, std::move(qmodel));

        WHEN("We create a state using the default value") {
            auto state = graph.empty_state();
            std::vector<double> vec_d{0, 1, 0, 1, 0};
            binary_node_ptr->initialize_state(state, vec_d);
            graph.initialize_state(state);

            THEN("We can read the state of the quadratic model node") {
                CHECK_THAT(qnode_ptr->view(state), RangeEquals({16}));
            }

            WHEN("We update the elements of the binary node and propagate") {
                // Change to {1, 1, 1, 1, 1}
                for (int i = 0; i < binary_node_ptr->size(); i++) {
                    binary_node_ptr->set(state, i);
                }

                binary_node_ptr->propagate(state);
                qnode_ptr->propagate(state);

                THEN("The state of the quadratic node is changed accordingly") {
                    CHECK_THAT(qnode_ptr->view(state), RangeEquals({70}));
                }

                WHEN("We commit previous changes and stack more changes") {
                    // Change to {0, 0, 0, 0, 0}
                    binary_node_ptr->commit(state);
                    for (int i = 0; i < binary_node_ptr->size(); i++) {
                        binary_node_ptr->unset(state, i);
                    }

                    binary_node_ptr->propagate(state);
                    qnode_ptr->propagate(state);

                    THEN("The state of the quadratic node is changed accordingly") {
                        CHECK_THAT(qnode_ptr->view(state), RangeEquals({0}));
                    }
                }
            }

            WHEN("We update the binary node many times with relatively small effective change") {
                // Change to {1, 1, 1, 1, 1}
                for (int i = 0; i < binary_node_ptr->size(); i++) {
                    binary_node_ptr->set(state, i);
                    binary_node_ptr->unset(state, i);
                    binary_node_ptr->flip(state, i);
                    binary_node_ptr->set(state, i);
                }

                binary_node_ptr->propagate(state);
                qnode_ptr->propagate(state);

                THEN("The number of updates in the binary node is longer than its length") {
                    CHECK(binary_node_ptr->diff(state).size() >
                          static_cast<size_t>(binary_node_ptr->size()));
                }

                THEN("The state of the quadratic node is changed accordingly") {
                    CHECK_THAT(qnode_ptr->view(state), RangeEquals({70}));
                }
            }
        }
    }

    GIVEN("A QuadraticModelNode with a list variables of length 5") {
        const int num_elements = 5;
        auto list_node_ptr = graph.emplace_node<ListNode>(num_elements);
        auto qnode_ptr = graph.emplace_node<QuadraticModelNode>(list_node_ptr, std::move(qmodel));

        WHEN("We create a state using the default value") {
            auto state = graph.empty_state();
            graph.initialize_state(state);

            THEN("We can read the state of the quadratic model node") {
                CHECK_THAT(qnode_ptr->view(state), RangeEquals({430}));
            }

            WHEN("We update the elements of the list node and propagate") {
                // Change to {1, 2, 3, 4, 0}
                for (int i = 0; i < list_node_ptr->size() - 1; i++) {
                    list_node_ptr->exchange(state, i, i + 1);
                }

                list_node_ptr->propagate(state);
                qnode_ptr->propagate(state);

                THEN("The state of the quadratic node is changed accordingly") {
                    CHECK_THAT(qnode_ptr->view(state), RangeEquals({290}));
                }

                WHEN("We commit previous changes and stack more changes") {
                    list_node_ptr->commit(state);

                    // Change to {0, 1, 2, 3, 4}
                    for (int i = list_node_ptr->size() - 1; i > 0; i--) {
                        list_node_ptr->exchange(state, i, i - 1);
                    }

                    list_node_ptr->propagate(state);
                    qnode_ptr->propagate(state);

                    THEN("The state of the quadratic node is changed accordingly") {
                        CHECK_THAT(qnode_ptr->view(state), RangeEquals({430}));
                    }
                }
            }

            WHEN("We update the binary node many times with relatively small effective change") {
                // Change to {0, 1, 2, 3, 4}
                for (int i = 0; i < list_node_ptr->size() - 1; i++) {
                    list_node_ptr->exchange(state, i, i + 1);
                }
                for (int i = list_node_ptr->size() - 1; i > 0; i--) {
                    list_node_ptr->exchange(state, i, i - 1);
                }
                for (int i = 0; i < list_node_ptr->size() - 1; i++) {
                    list_node_ptr->exchange(state, i, i + 1);
                }
                for (int i = list_node_ptr->size() - 1; i > 0; i--) {
                    list_node_ptr->exchange(state, i, i - 1);
                }

                list_node_ptr->propagate(state);
                qnode_ptr->propagate(state);

                THEN("The number of updates in the binary node is longer than its length") {
                    CHECK(list_node_ptr->diff(state).size() >
                          static_cast<size_t>(list_node_ptr->size()));
                }

                THEN("The state of the quadratic node is changed accordingly") {
                    CHECK_THAT(qnode_ptr->view(state), RangeEquals({430}));
                }
            }
        }
    }
}

}  // namespace dwave::optimization

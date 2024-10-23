// Copyright 2023 D-Wave Systems Inc.
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

#include <unordered_set>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes.hpp"

namespace dwave::optimization {

TEST_CASE("Knapsack problem", "[Knapsack]") {
    const int num_items = 4;
    const std::vector<double> weights = {30, 10, 40, 20};
    const std::vector<double> values = {-10, -20, -30, -40};
    const int capacity = 40;

    GIVEN("A graph representing the knapsack problem") {
        auto graph = Graph();

        // These are copies. It's actually possible to do it without the copy
        // but let's keep it simple for now
        auto weights_ptr = graph.emplace_node<ConstantNode>(weights);
        auto values_ptr = graph.emplace_node<ConstantNode>(values);

        auto items_ptr = graph.emplace_node<SetNode>(num_items);  // decision

        auto item_weights_ptr = graph.emplace_node<AdvancedIndexingNode>(weights_ptr, items_ptr);
        auto item_values_ptr = graph.emplace_node<AdvancedIndexingNode>(values_ptr, items_ptr);

        auto weight_ptr = graph.emplace_node<SumNode>(item_weights_ptr);
        auto value_ptr = graph.emplace_node<SumNode>(item_values_ptr);

        auto capacity_ptr = graph.emplace_node<ConstantNode>(capacity);

        auto le_ptr = graph.emplace_node<LessEqualNode>(weight_ptr, capacity_ptr);

        graph.set_objective(value_ptr);
        graph.add_constraint(le_ptr);

        WHEN("We default-initialize a state") {
            auto state = graph.initialize_state();

            THEN("We can inspect the state of each of the nodes") {
                CHECK(std::ranges::equal(weights_ptr->view(state), weights));
                CHECK(std::ranges::equal(values_ptr->view(state), values));

                // assume it gets default-initialized to empty, other tests rely
                // on this assumption
                REQUIRE(std::ranges::equal(items_ptr->view(state), std::vector<double>{}));

                CHECK(std::ranges::equal(item_weights_ptr->view(state), std::vector<double>{}));
                CHECK(std::ranges::equal(item_values_ptr->view(state), std::vector<double>{}));

                CHECK(std::ranges::equal(weight_ptr->view(state), std::vector{0}));
                CHECK(std::ranges::equal(value_ptr->view(state), std::vector{0}));

                CHECK(std::ranges::equal(le_ptr->view(state), std::vector{true}));  // 0 <= 40

                CHECK(graph.energy(state) == 0);
                CHECK(graph.feasible(state));
            }

            AND_WHEN("We update and propagate the state manually") {
                items_ptr->grow(state);
                items_ptr->propagate(state);

                THEN("items are updated accordingly") {
                    CHECK(std::ranges::equal(items_ptr->view(state), std::vector{0}));
                }

                item_weights_ptr->propagate(state);
                item_values_ptr->propagate(state);

                THEN("The weights/values are updated accordingly") {
                    CHECK(std::ranges::equal(item_weights_ptr->view(state), std::vector{30}));
                    CHECK(std::ranges::equal(item_values_ptr->view(state), std::vector{-10}));
                }

                weight_ptr->propagate(state);
                value_ptr->propagate(state);

                THEN("The weight/value are updated accordingly") {
                    CHECK(std::ranges::equal(weight_ptr->view(state), std::vector{30}));
                    CHECK(std::ranges::equal(value_ptr->view(state), std::vector{-10}));
                }

                le_ptr->propagate(state);

                THEN("The violation is updated accordingly") {
                    CHECK(std::ranges::equal(le_ptr->view(state), std::vector{true}));  // 30 <= 40
                }

                THEN("The energy and feasibility are what we expect") {
                    CHECK(graph.energy(state) == -10);
                    CHECK(graph.feasible(state));
                }
            }
        }

        WHEN("We do random moves and randomly reject them") {
            constexpr int num_moves = 100;

            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            for (int i = 0; i < num_moves; ++i) {
                // do a move
                items_ptr->default_move(state, rng);
                auto changed_nodes = std::vector<const Node*>{items_ptr};

                graph.propose(state, changed_nodes, [&rng](const Graph& graph, State& state) {
                    std::uniform_int_distribution<int> r(0, 1);
                    return r(rng);
                });

                // check that the state is a subset of the range
                std::unordered_set<int> set(items_ptr->begin(state), items_ptr->end(state));
                CHECK(static_cast<ssize_t>(set.size()) == items_ptr->size(state));
                CHECK(*std::min_element(items_ptr->begin(state), items_ptr->end(state)) >= 0);
                CHECK(*std::max_element(items_ptr->begin(state), items_ptr->end(state)) <
                      num_items);

                // check that the calculated energy matches the "by hand" calculation
                double en = 0;
                {
                    for (const auto& item : items_ptr->view(state)) {
                        en += values[item];
                    }
                }
                CHECK(en == graph.energy(state));

                double w = 0;
                for (const auto& item : items_ptr->view(state)) {
                    w += weights[item];
                }
                CHECK((w <= capacity) == graph.feasible(state));
            }
        }
    }
}

TEST_CASE("Capacitated Multi-Vehicle Routing Problem", "[CVRP]") {
    auto rng = dwave::optimization::RngAdaptor(std::mt19937(42));

    const size_t num_customers = 30;
    const size_t num_vehicles = 4;
    const size_t capacity_val = 15;

    std::vector<double> customer_demands_data(num_customers, 1.0);

    std::uniform_int_distribution<int> random_loc(-20, 20);
    std::vector<int> customer_locations_x(num_customers);
    std::vector<int> customer_locations_y(num_customers);
    for (size_t i = 0; i < num_customers; i++) {
        customer_locations_x[i] = random_loc(rng);
        customer_locations_y[i] = random_loc(rng);
    }

    double* distance_matrix = new double[num_customers * num_customers];
    double* depot_distances = new double[num_customers];
    for (size_t i = 0; i < num_customers; i++) {
        for (size_t j = 0; j < num_customers; j++) {
            int x1 = customer_locations_x[i];
            int y1 = customer_locations_y[i];
            int x2 = customer_locations_x[j];
            int y2 = customer_locations_y[j];
            double distance = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
            distance_matrix[i * num_customers + j] = distance;
        }
        depot_distances[i] = std::sqrt(std::pow(customer_locations_x[i], 2) +
                                       std::pow(customer_locations_y[i], 2));
    }

    GIVEN("A graph representing the CVRP") {
        auto model = dwave::optimization::Graph();

        auto route_dl = model.emplace_node<dwave::optimization::DisjointListsNode>(num_customers,
                                                                                   num_vehicles);
        std::vector<dwave::optimization::DisjointListNode*> routes;
        for (size_t i = 0; i < num_vehicles; i++) {
            routes.push_back(model.emplace_node<dwave::optimization::DisjointListNode>(route_dl));
        }

        auto D = model.emplace_node<dwave::optimization::ConstantNode>(
                distance_matrix, std::initializer_list<ssize_t>{num_customers, num_customers});
        auto depot_D = model.emplace_node<dwave::optimization::ConstantNode>(
                depot_distances, std::initializer_list<ssize_t>{num_customers});

        auto customer_demands = model.emplace_node<dwave::optimization::ConstantNode>(
                customer_demands_data.data(), std::initializer_list<ssize_t>{num_customers});

        auto capacity = model.emplace_node<dwave::optimization::ConstantNode>(capacity_val);

        std::vector<dwave::optimization::ArrayNode*> sum_d_partition;
        std::vector<dwave::optimization::ArrayNode*> depot_sums_a;
        std::vector<dwave::optimization::ArrayNode*> depot_sums_b;

        for (size_t i = 0; i < num_vehicles; i++) {
            auto first_cust = model.emplace_node<dwave::optimization::BasicIndexingNode>(
                    routes[i], dwave::optimization::Slice(1));
            auto first_cust_d = model.emplace_node<dwave::optimization::AdvancedIndexingNode>(
                    depot_D, first_cust);
            auto first_cust_d_sum = model.emplace_node<dwave::optimization::SumNode>(first_cust_d);
            depot_sums_a.push_back(first_cust_d_sum);

            auto last_cust = model.emplace_node<dwave::optimization::BasicIndexingNode>(
                    routes[i], dwave::optimization::Slice(-1, std::nullopt));
            auto last_cust_d = model.emplace_node<dwave::optimization::AdvancedIndexingNode>(
                    depot_D, last_cust);
            auto last_cust_d_sum = model.emplace_node<dwave::optimization::SumNode>(last_cust_d);
            depot_sums_b.push_back(last_cust_d_sum);

            auto starts = model.emplace_node<dwave::optimization::BasicIndexingNode>(
                    routes[i], dwave::optimization::Slice(0, -1));
            auto ends = model.emplace_node<dwave::optimization::BasicIndexingNode>(
                    routes[i], dwave::optimization::Slice(1, std::nullopt));
            auto d_customers =
                    model.emplace_node<dwave::optimization::AdvancedIndexingNode>(D, starts, ends);

            sum_d_partition.push_back(
                    model.emplace_node<dwave::optimization::SumNode>(d_customers));

            // Capacity constraint implemented here is not yet validated by this test
            auto demands = model.emplace_node<dwave::optimization::AdvancedIndexingNode>(
                    customer_demands, routes[i]);
            auto sum_demand_partition = model.emplace_node<dwave::optimization::SumNode>(demands);
            auto le_node = model.emplace_node<dwave::optimization::LessEqualNode>(
                    sum_demand_partition, capacity);
            model.add_constraint(le_node);
        }

        auto cust_ds = model.emplace_node<dwave::optimization::NaryAddNode>(sum_d_partition);
        auto depot_ds_a = model.emplace_node<dwave::optimization::NaryAddNode>(depot_sums_a);
        auto depot_ds_b = model.emplace_node<dwave::optimization::NaryAddNode>(depot_sums_b);

        auto tot_d = model.emplace_node<dwave::optimization::NaryAddNode>(cust_ds);
        tot_d->add_node(depot_ds_a);
        tot_d->add_node(depot_ds_b);

        model.set_objective(tot_d);

        WHEN("We do random moves") {
            auto state = model.initialize_state();

            for (size_t sweep = 0; sweep < 100; sweep++) {
                route_dl->default_move(state, rng);

                model.propose(state, {route_dl});

                {
                    double en = 0.0;
                    for (size_t route_idx = 0; route_idx < num_vehicles; route_idx++) {
                        auto route = routes[route_idx]->view(state);
                        auto route_size = routes[route_idx]->size(state);
                        if (route_size > 0) {
                            en += depot_distances[static_cast<size_t>(route[0])];
                            for (ssize_t i = 0; i < route_size - 1; i++) {
                                en += distance_matrix[static_cast<size_t>(route[i] * num_customers +
                                                                          route[i + 1])];
                            }
                            en += depot_distances[static_cast<size_t>(route[route_size - 1])];
                        }
                    }

                    REQUIRE_THAT(model.energy(state), Catch::Matchers::WithinRel(en, 0.00001));
                }
            }
        }

        WHEN("We do random moves and randomly reject them") {
            auto state = model.initialize_state();

            for (size_t sweep = 0; sweep < 100; sweep++) {
                route_dl->default_move(state, rng);

                model.propose(state, {route_dl},
                              [&rng](const dwave::optimization::Graph& model,
                                     dwave::optimization::State& state) {
                                  std::uniform_int_distribution<int> r(0, 1);
                                  return r(rng);
                              });

                {
                    double en = 0.0;
                    for (size_t route_idx = 0; route_idx < num_vehicles; route_idx++) {
                        auto route = routes[route_idx]->view(state);
                        auto route_size = routes[route_idx]->size(state);
                        if (route_size > 0) {
                            en += depot_distances[static_cast<size_t>(route[0])];
                            for (ssize_t i = 0; i < route_size - 1; i++) {
                                en += distance_matrix[static_cast<size_t>(route[i] * num_customers +
                                                                          route[i + 1])];
                            }
                            en += depot_distances[static_cast<size_t>(route[route_size - 1])];
                        }
                    }

                    REQUIRE_THAT(model.energy(state), Catch::Matchers::WithinRel(en, 0.00001));
                }
            }
        }
    }
}

TEST_CASE("Quadratic Assignment Problem", "[QAP]") {
    const int num_items = 5;

    // 5x5 flow matrix - not symmetric
    // [[ 0  8  7  4  4]
    //  [ 9  0  7  2  1]
    //  [ 5 10  0  8  7]
    //  [ 8  5  1  0  4]
    //  [ 5  4  2 10  0]]
    const std::vector<int> W = {0, 8, 7, 4, 4, 9, 0, 7, 2, 1, 5,  10, 0,
                                8, 7, 8, 5, 1, 0, 4, 5, 4, 2, 10, 0};

    // 5x5 distance matrix - symmetric
    // [[0 5 9 6 5]
    //  [5 0 1 6 9]
    //  [9 1 0 3 7]
    //  [6 6 3 0 1]
    //  [5 9 7 1 0]]
    const std::vector<int> D = {0, 5, 9, 6, 5, 5, 0, 1, 6, 9, 9, 1, 0,
                                3, 7, 6, 6, 3, 0, 1, 5, 9, 7, 1, 0};

    GIVEN("A graph representing a quadratic assignment problem") {
        auto graph = Graph();

        // x[i] is the facility at location i
        auto x_ptr = graph.emplace_node<ListNode>(num_items);

        auto W_ptr = graph.emplace_node<ConstantNode>(
                W, std::initializer_list<ssize_t>{num_items, num_items});
        auto D_ptr = graph.emplace_node<ConstantNode>(
                D, std::initializer_list<ssize_t>{num_items, num_items});

        // D[x, :][:, x]
        auto xDx_ptr = graph.emplace_node<PermutationNode>(D_ptr, x_ptr);

        auto costs_ptr = graph.emplace_node<MultiplyNode>(W_ptr, xDx_ptr);
        auto cost_ptr = graph.emplace_node<SumNode>(costs_ptr);

        graph.set_objective(cost_ptr);

        THEN("The shape of everything is known at model construction time") {
            CHECK(std::ranges::equal(x_ptr->shape(), std::vector{num_items}));
            CHECK(std::ranges::equal(W_ptr->shape(), std::vector{num_items, num_items}));
            CHECK(std::ranges::equal(D_ptr->shape(), std::vector{num_items, num_items}));
            CHECK(std::ranges::equal(xDx_ptr->shape(), std::vector{num_items, num_items}));
        }

        WHEN("We default-initialize a state") {
            auto state = graph.initialize_state();

            THEN("We can inspect the state of each of the nodes") {
                // x gets default initialized to a range
                CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 1, 2, 3, 4}));

                CHECK(std::ranges::equal(W_ptr->view(state), W));
                CHECK(std::ranges::equal(D_ptr->view(state), D));

                CHECK(std::ranges::equal(xDx_ptr->view(state), D));
            }
        }

        WHEN("We initialize the state") {
            auto state = graph.empty_state();
            x_ptr->initialize_state(state, std::vector<double>{0, 2, 1, 3, 4});
            graph.initialize_state(state);

            THEN("We can inspect the state of each of the nodes") {
                CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 2, 1, 3, 4}));

                CHECK(std::ranges::equal(W_ptr->view(state), W));
                CHECK(std::ranges::equal(D_ptr->view(state), D));

                // [[0 9 5 6 5]
                //  [9 0 1 3 7]
                //  [5 1 0 6 9]
                //  [6 3 6 0 1]
                //  [5 7 9 1 0]]
                CHECK(std::ranges::equal(xDx_ptr->view(state),
                                         std::vector{0, 9, 5, 6, 5, 9, 0, 1, 3, 7, 5, 1, 0,
                                                     6, 9, 6, 3, 6, 0, 1, 5, 7, 9, 1, 0}));

                CHECK(cost_ptr->view(state).front() == 552);

                CHECK(graph.energy(state) == 552);
            }

            AND_WHEN("We update and propagate") {
                x_ptr->exchange(state, 2, 3);  // [0, 2, 3, 1, 4]
                x_ptr->propagate(state);

                CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 2, 3, 1, 4}));

                xDx_ptr->propagate(state);

                // [[0 9 6 5 5]
                //  [9 0 3 1 7]
                //  [6 3 0 6 1]
                //  [5 1 6 0 9]
                //  [5 7 1 9 0]]
                CHECK(std::ranges::equal(xDx_ptr->view(state),
                                         std::vector{0, 9, 6, 5, 5, 9, 0, 3, 1, 7, 6, 3, 0,
                                                     6, 1, 5, 1, 6, 0, 9, 5, 7, 1, 9, 0}));

                costs_ptr->propagate(state);
                cost_ptr->propagate(state);

                CHECK(cost_ptr->view(state).front() == 612);
            }
        }

        WHEN("We do random moves") {
            constexpr int num_moves = 10;

            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            for (int i = 0; i < num_moves; ++i) {
                // do a move
                x_ptr->default_move(state, rng);
                auto changed_nodes = std::vector<const Node*>{x_ptr};

                graph.propose(state, changed_nodes);

                // check that the calculated energy matches the "by hand" calculation
                double en = 0;
                {
                    auto x = x_ptr->view(state);

                    for (int a = 0; a < num_items; ++a) {
                        for (int b = 0; b < num_items; ++b) {
                            en += W.at(a * num_items + b) * D.at(x[a] * num_items + x[b]);
                        }
                    }
                }
                CHECK(en == graph.energy(state));
            }
        }

        WHEN("We do random moves and randomly reject them") {
            constexpr int num_moves = 10;

            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            for (int i = 0; i < num_moves; ++i) {
                // do a move
                x_ptr->default_move(state, rng);
                auto changed_nodes = std::vector<const Node*>{x_ptr};

                graph.propose(state, changed_nodes, [&rng](const Graph& graph, State& state) {
                    std::uniform_int_distribution<int> r(0, 1);
                    return r(rng);
                });

                // check that the calculated energy matches the "by hand" calculation
                double en = 0;
                {
                    auto x = x_ptr->view(state);

                    for (int a = 0; a < num_items; ++a) {
                        for (int b = 0; b < num_items; ++b) {
                            en += W.at(a * num_items + b) * D.at(x[a] * num_items + x[b]);
                        }
                    }
                }
                CHECK(en == graph.energy(state));
            }
        }
    }
}

TEST_CASE("Traveling Salesperson Problem", "[TSP]") {
    const int num_cities = 5;

    // 5x5 distance matrix - symmetric
    // [[0 5 9 6 5]
    //  [5 0 1 6 9]
    //  [9 1 0 3 7]
    //  [6 6 3 0 1]
    //  [5 9 7 1 0]]
    const std::vector<int> D = {0, 5, 9, 6, 5, 5, 0, 1, 6, 9, 9, 1, 0,
                                3, 7, 6, 6, 3, 0, 1, 5, 9, 7, 1, 0};

    GIVEN("A graph representing a traveling salesperson problem") {
        auto graph = Graph();

        auto D_ptr = graph.emplace_node<ConstantNode>(
                D, std::initializer_list<ssize_t>{num_cities, num_cities});

        // x[i] is the city visted at time i
        auto x_ptr = graph.emplace_node<ListNode>(num_cities);

        // x[:-1]
        auto starts_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(0, -1));
        // x[1:]
        auto ends_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt));

        // x[0]
        auto start_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, 0);

        // x[-1]
        auto end_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, -1);

        // D[x[:-1], x[1:]]
        auto path_ptr = graph.emplace_node<AdvancedIndexingNode>(D_ptr, starts_ptr, ends_ptr);

        // D[x[-1], x[0]]
        auto link_ptr = graph.emplace_node<AdvancedIndexingNode>(D_ptr, end_ptr, start_ptr);

        // D[x[:-1], x[1:]].sum()
        auto path_sum_ptr = graph.emplace_node<SumNode>(path_ptr);

        // D[x[:-1], x[1:]].sum() + D[x[-1], x[0]]
        auto sum_ptr = graph.emplace_node<AddNode>(path_sum_ptr, link_ptr);

        graph.set_objective(sum_ptr);

        WHEN("We initialize the state") {
            auto state = graph.initialize_state();

            THEN("We see the initial values we expect") {
                CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 1, 2, 3, 4}));
                CHECK(std::ranges::equal(starts_ptr->view(state), std::vector{0, 1, 2, 3}));
                CHECK(std::ranges::equal(ends_ptr->view(state), std::vector{1, 2, 3, 4}));
                CHECK(std::ranges::equal(start_ptr->view(state), std::vector{0}));
                CHECK(std::ranges::equal(end_ptr->view(state), std::vector{4}));
                CHECK(std::ranges::equal(path_ptr->view(state), std::vector{5, 1, 3, 1}));
                CHECK(std::ranges::equal(link_ptr->view(state), std::vector{5}));
                CHECK(std::ranges::equal(sum_ptr->view(state), std::vector{15}));
            }

            AND_WHEN("We swap two cities in the route and then propagate") {
                x_ptr->exchange(state, 1, 3);
                graph.propose(state, {x_ptr});

                THEN("We see the updated state we expect") {
                    CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 3, 2, 1, 4}));
                    CHECK(std::ranges::equal(starts_ptr->view(state), std::vector{0, 3, 2, 1}));
                    CHECK(std::ranges::equal(ends_ptr->view(state), std::vector{3, 2, 1, 4}));
                    CHECK(std::ranges::equal(start_ptr->view(state), std::vector{0}));
                    CHECK(std::ranges::equal(end_ptr->view(state), std::vector{4}));
                    CHECK(std::ranges::equal(path_ptr->view(state), std::vector{6, 3, 1, 9}));
                    CHECK(std::ranges::equal(link_ptr->view(state), std::vector{5}));
                    CHECK(std::ranges::equal(sum_ptr->view(state), std::vector{24}));
                }
            }
        }

        WHEN("We do random moves and always accept") {
            constexpr int num_sweeps = 50;

            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            for (int i = 0; i < num_sweeps; ++i) {
                // do a move
                x_ptr->default_move(state, rng);
                graph.propose(state, {x_ptr});

                // check that the calculated energy matches the "by hand" calculation
                double en = 0;
                {
                    auto x = x_ptr->view(state);

                    for (int city = 0; city < num_cities - 1; ++city) {
                        en += D[x[city] * num_cities + x[city + 1]];
                    }
                    en += D[x[num_cities - 1] * num_cities + x[0]];
                }
                CHECK(en == graph.energy(state));
            }
        }
        WHEN("We do random moves and randomly accept") {
            constexpr int num_sweeps = 50;

            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            for (int i = 0; i < num_sweeps; ++i) {
                // do a move
                x_ptr->default_move(state, rng);

                graph.propose(state, {x_ptr}, [&rng](const Graph& graph, State& state) {
                    std::uniform_int_distribution<int> r(0, 1);
                    return r(rng);
                });

                // check that the calculated energy matches the "by hand" calculation
                double en = 0;
                {
                    auto x = x_ptr->view(state);

                    for (int city = 0; city < num_cities - 1; ++city) {
                        en += D[x[city] * num_cities + x[city + 1]];
                    }
                    en += D[x[num_cities - 1] * num_cities + x[0]];
                }
                CHECK(en == graph.energy(state));
            }
        }
    }
}

}  // namespace dwave::optimization

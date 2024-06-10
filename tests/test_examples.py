# Copyright 2024 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import unittest

from functools import reduce
import numpy as np

from dwave.optimization import Model, add, logical_or, logical_and, maximum


class TestCVRP(unittest.TestCase):
    def test_cvrp(self):
        # Problem definition
        num_customers = 30
        num_vehicles = 4
        capacity_val = 15
        customer_demands = np.array([1 for _ in range(num_customers)])
        customer_locations_x = np.random.randint(-20, 20, size=num_customers)
        customer_locations_y = np.random.randint(-20, 20, size=num_customers)

        x0, x1 = np.meshgrid(customer_locations_x, customer_locations_x)
        y0, y1 = np.meshgrid(customer_locations_y, customer_locations_y)
        distance_matrix = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        depot_distance_vector = np.sqrt(
            customer_locations_x ** 2 + customer_locations_y ** 2
        )

        # Now build the model, using a single disjoint lists variable
        model = Model()

        D = model.constant(distance_matrix)
        depot_D = model.constant(depot_distance_vector)
        demands = model.constant(customer_demands)
        capacity = model.constant(capacity_val)

        routes_decision, routes = model.disjoint_lists(
            primary_set_size=num_customers,
            num_disjoint_lists=num_vehicles
        )

        route_costs = []
        for r in range(num_vehicles):
            route_costs.append(depot_D[routes[r][:1]].sum())
            route_costs.append(depot_D[routes[r][-1:]].sum())
            route_costs.append(D[routes[r][1:], routes[r][:-1]].sum())

            model.add_constraint(demands[routes[r]].sum() <= capacity)

        model.minimize(add(route_costs))

        model.states.resize(1)
        model.lock()

        # Now test that the energy is correct with an example set of routes
        example_routes = [
            list(range(0, 10)),
            list(range(10, 20)),
            list(range(20, 25)),
            list(range(25, 30)),
        ]
        routes_decision.set_state(0, example_routes)

        # Calculate the energy manually
        en = 0.0
        for r in range(num_vehicles):
            route = example_routes[r]
            if len(route) == 0:
                continue
            en += depot_distance_vector[route[0]] + depot_distance_vector[route[-1]]
            en += distance_matrix[route[:-1], route[1:]].sum()

        self.assertAlmostEqual(en, model.objective.state(0))

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())


class TestFlowShopProblem(unittest.TestCase):

    def test_2x2(self):

        n_jobs = 2
        n_machines = 2
        processing_times = np.array([[10, 5], [20, 10]])

        model = Model()
        pt = model.constant(processing_times)
        order = model.list(n_jobs)

        end_times = []
        for m in range(n_machines):

            this_times = []
            if m == 0:

                for j in range(n_jobs):
                    if j == 0:
                        this_times.append(pt[m, :][order[j]])
                    else:
                        et = pt[m, :][order[j]].sum()
                        et += this_times[-1].sum()
                        this_times.append(et)

            else:

                for j in range(n_jobs):
                    if j == 0:
                        et = end_times[m - 1][j].sum()
                        et += pt[m, :][order[j]].sum()
                        this_times.append(et)
                    else:
                        et = maximum(end_times[m - 1][j], this_times[-1]).sum()
                        et += pt[m, :][order[j]].sum()
                        this_times.append(et)

            end_times.append(this_times)

        ms = end_times[-1][-1]
        model.minimize(ms)

        model.lock()
        model.states.resize(1)

        np.testing.assert_array_equal(order.state(0), range(2))
        self.assertEqual(end_times[1][0].state(0), 30)
        self.assertEqual(end_times[1][1].state(0), 40)
        self.assertEqual(ms.state(0), 40)

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())


class TestJobShopProblem_Formulation1(unittest.TestCase):
    def create_test_model(self, T_in, J_in, M_in, X_in, makespan_expected):
        # Ref: https://hal-lirmm.ccsd.cnrs.fr/LAAS-ROC/tel-03923149v2
        model = Model()

        n_j = int(max(T_in[:,1]))
        n_m = int(max(T_in[:,2]))
        n_t = len(T_in)

        machine = model.constant(M_in)

        start = model.integer(n_t, upper_bound=sum(T_in[:,0]))
        duration = model.constant(T_in[:,0])
        end = start + duration

        n_c = 0
        for j in J_in:
            for i in range(1, len(j)):
                n_c += 1
                model.add_constraint(
                    end[j[i-1]] <= start[j[i]]
                    )
                
        for m in range(0, n_m):
            order = model.list(n_j)
            for j in range(0,n_j-1):
                n_c += 1
                model.add_constraint(
                    end[machine[m,:][order[j]]] <= start[machine[m,:][order[j+1]]]
                    )

        obj_func = end.max()
        model.minimize( obj_func )

        model.states.resize(1)
        model.lock()

        start.set_state(0, X_in)

        makespan_actual = obj_func.state(0)
        self.assertEqual(makespan_expected, makespan_actual)

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())

    def test_small_3jobs_3tasks(self):
        T_in = np.array([ (3,0,0), (4,0,2), (2,0,1), 
                          (4,1,1), (3,1,0), (2,1,2),
                          (2,2,0), (2,2,2), (3,2,1),
                          ])
        J_in = np.array([
            [ 0, 1, 2 ],
            [ 3, 4, 5 ],
            [ 6, 7, 8 ],
        ])
        M_in = np.array([
            [ 0, 5, 7 ],
            [ 2, 3, 8 ],
            [ 1, 4, 6 ]
        ])
        # M0: 00077444..
        # M1: 3333.88822
        # M2: 66.1111.55
        X_in = [ 0, 3, 8, 0, 5, 8, 0, 3, 5 ]
        makespan_expected = 10

        self.create_test_model(T_in, J_in, M_in, X_in, makespan_expected)


class TestJobShopProblem_Formulation2(unittest.TestCase):
    def create_test_model(self, T_in, J_in, M_in, X_in, makespan_expected):
        model = Model()

        n_j = max(T_in[:,1])
        n_m = max(T_in[:,2])
        n_t = len(T_in)

        # T = model.constant(T_in)
        C = model.constant(T_in[:,0])
        
        # $J$ is the set of sets of tasks, where $J_i$ is the {\em execution-ordered} set of 
        # tasks for the $i$-th job
        # J = model.constant(J_in)
        
        # $M$ is the set of sets of tasks, where $M_i$ is the unordered set of tasks for 
        # the $i$-th machine
        # M = model.constant(M_in)

        # $X$ is a vector of integer variables, where $x_t$ is the time that task $t$ is 
        # scheduled to start
        X = model.integer(n_t, upper_bound=sum(T_in[:,0]))
        self.assertEqual(1, model.num_decisions())

        # Constraint to ensure that all of a job's tasks occur in the correct order:
        # $x_{j_{i-1}} + c_{j_{i-1}} <= x_{j_i}, \forall j \in J, 1 < i <= |j|$ 
        n_c = 0
        for j in J_in:
            for i in range(1, len(j)):
                if j[i] is not None:
                    n_c += 1
                    model.add_constraint(X[j[i-1]] + C[j[i-1]] <= X[j[i]])

        # Constraint to ensure that all of a machine's tasks do /not/ overlap:
        # $x_{m_i} + c_{m_i} <= x_{m_j} \vee x_{m_j} + c_{m_j} < x_{m_i}, \forall i,j \in M, i \neq j$ 
        for m in M_in:
            for i in range(0,len(m)):
                for j in range(0,len(m)):
                    if i != j and m[i] is not None and m[j] is not None:
                        n_c += 1
                        model.add_constraint(
                            logical_or(
                                X[m[i]] + C[m[i]] <= X[m[j]],
                                X[m[j]] + C[m[j]] <= X[m[i]]
                            ))

        self.assertEqual(n_c, model.num_constraints())

        # Objective function:
        # minimize $f(X) = \max (X+C) - \min X$ 
        obj_func = (X+C).max() - X.min()
        model.minimize( obj_func )

        model.states.resize(1)
        model.lock()

        X.set_state(0, X_in)

        makespan_actual = obj_func.state(0)
        self.assertEqual(makespan_expected, makespan_actual)

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())

    def test_small_2jobs_3tasks(self):
        # $T$ is the set of all tasks, where a task $t$ is a tuple $(c,j,m)$ where $c_t$ is its 
        # cost, $j_t$ is the job it belongs to, and $m_t$ is the machine that it needs to be run on
        T_in = np.array([ (3,0,0), (2,0,1), (2,0,2), (2,1,0), (1,1,2), (4,1,1) ])
        J_in = np.array([
            [ 0, 1, 2 ],
            [ 3, 4, 5 ],
        ])
        M_in = np.array([
            [ 0, 3 ],
            [ 1, 5 ],
            [ 2, 4 ]
        ])
        # M0: 33000
        # M1: .....115555
        # M2: ..4....22
        X_in = [ 2, 5, 7, 0, 2, 7 ]
        makespan_expected = 11

        self.create_test_model(T_in, J_in, M_in, X_in, makespan_expected)

    def test_small_3jobs_3tasks(self):
        # $T$ is the set of all tasks, where a task $t$ is a tuple $(c,j,m)$ where $c_t$ is its 
        # cost, $j_t$ is the job it belongs to, and $m_t$ is the machine that it needs to be run on
        T_in = np.array([ (3,0,0), (4,0,2), (2,0,1), 
                          (4,1,1), (3,1,0), (2,1,2),
                          (2,2,0), (2,2,2), (3,2,1),
                          ])
        J_in = np.array([
            [ 0, 1, 2 ],
            [ 3, 4, 5 ],
            [ 6, 7, 8 ],
        ])
        M_in = np.array([
            [ 0, 5, 7 ],
            [ 2, 3, 8 ],
            [ 1, 4, 6 ]
        ])
        # M0: 00077444..
        # M1: 3333.88822
        # M2: 66.1111.55
        X_in = [ 0, 3, 8, 0, 5, 8, 0, 3, 5 ]
        makespan_expected = 10

        self.create_test_model(T_in, J_in, M_in, X_in, makespan_expected)

    def test_small_3jobs_ragged_padded(self):
        # Ref: https://developers.google.com/optimization/scheduling/job_shop
        T_in = np.array([ (3,0,0), (2,0,1), (2,0,2), (2,1,0), (1,1,2), (4,1,1), (4,2,1), (3,2,2) ])
        J_in = np.array([
            [ 0, 1, 2 ],
            [ 3, 4, 5 ],
            [ 6, 7, None ]
        ])
        M_in = np.array([
            [ 0, 3, None ],
            [ 1, 5, 6 ],
            [ 2, 4, 7 ]
        ])
        X_in = [ 2, 8, 10, 0, 2, 4, 0, 4 ]
        makespan_expected = 12

        self.create_test_model(T_in, J_in, M_in, X_in, makespan_expected)


class TestKnapsackProblem(unittest.TestCase):

    def test_4_items(self):

        n_items = 4
        model = Model()
        weights = model.constant([30, 10, 40, 20])
        values = model.constant([-10, -20, -30, -40])

        # need to add `sum()` because we don't support broadcasting
        # for the constraint yet
        capacity = model.constant(40).sum()
        items = model.set(n_items)

        # add the capacity constraint
        le = weights[items].sum() <= capacity
        model.add_constraint(le)

        # add the objective
        model.minimize(values[items].sum())

        model.lock()

        self.assertEqual(model.num_constraints(), 1)

        model.states.resize(1)

        items.set_state(0, [0, 1, 2, 3])
        np.testing.assert_array_equal(items.state(0), range(n_items))

        # check that the values are correct
        self.assertEqual(model.objective.state(0), -100)

        # check that the capacity "constraint" is violated
        self.assertEqual(le.state(0), 0)

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())

    def test_serialization(self):
        n_items = 4
        model = Model()
        weights = model.constant([30, 10, 40, 20])
        values = model.constant([-10, -20, -30, -40])

        # need to add `sum()` because we don't support broadcasting
        # for the constraint yet
        capacity = model.constant(40).sum()
        items = model.set(n_items)

        # add the capacity constraint
        le = weights[items].sum() <= capacity
        model.add_constraint(le)

        # add the objective
        model.minimize(values[items].sum())

        model.lock()

        with model.to_file() as f:
            new = Model.from_file(f)

        # todo: use a model equality test function once we have it
        for n0, n1 in zip(model.iter_symbols(), new.iter_symbols()):
            self.assertIs(type(n0), type(n1))

        self.assertIs(type(new.objective), type(model.objective))
        self.assertEqual(model.num_constraints(), new.num_constraints())

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())


class TestQuadraticAssignmentProblem(unittest.TestCase):
    def test_5x5(self):
        model = Model()

        # Flows - not symmetric
        W = model.constant(np.arange(25).reshape((5, 5)))

        # Distances - symmetric
        D = model.constant([[0, 10, 11, 12, 13],
                            [10, 0, 14, 15, 16],
                            [11, 14, 0, 17, 18],
                            [12, 15, 17, 0, 19],
                            [13, 16, 18, 19, 0]])

        # The assignments
        x = model.list(5)

        model.minimize((W * D[x, :][:, x]).sum())
        
        model.lock()

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())

    def test_serialization(self):
        model = Model()

        # Flows - not symmetric
        W = model.constant(np.arange(25).reshape((5, 5)))

        # Distances - symmetric
        D = model.constant([[0, 10, 11, 12, 13],
                            [10, 0, 14, 15, 16],
                            [11, 14, 0, 17, 18],
                            [12, 15, 17, 0, 19],
                            [13, 16, 18, 19, 0]])

        # The assignments
        x = model.list(5)

        model.minimize((W * D[x, :][:, x]).sum())

        model.lock()
        with model.to_file() as f:
            new = Model.from_file(f)

        # todo: use a model equality test function once we have it
        for n0, n1 in zip(model.iter_symbols(), new.iter_symbols()):
            self.assertIs(type(n0), type(n1))

        self.assertIs(type(new.objective), type(model.objective))

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())


class TestTravellingSalesperson(unittest.TestCase):
    def test_5x5(self):
        model = Model()

        D = model.constant([[0, 1, 2, 3, 4],
                            [1, 0, 5, 6, 7],
                            [2, 5, 0, 8, 9],
                            [3, 6, 8, 0, 10],
                            [4, 7, 9, 10, 0]])

        x = model.list(5)

        path = D[x[:-1], x[1:]]
        link = D[x[-1], x[0]]

        # We don't yet handle path.sum() + link.sum()
        cost = path.sum()
        cost += link.sum()

        model.minimize(cost)

        model.lock()

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())


class TestNurseScheduling(unittest.TestCase):
    def test_small(self):
        #####################################
        # Nurse scheduling problem definition
        num_days = 7
        num_nurses = 5
        num_shifts = 3
        max_shifts_in_a_row = 3
        max_minutes_per_week = 480 * 6
        min_minutes_per_week = 480 * 3

        # How many minutes each shift is (all 8 hours for now)
        minutes_per_shift_data = [480] * num_shifts
        #####################################

        # Build a model that implements this problem
        model = Model()

        minutes_per_shift = [
            model.constant(np.repeat(minutes, num_nurses))
            for minutes in minutes_per_shift_data
        ]

        bit_sets = [
            model.disjoint_bit_sets(num_nurses, num_shifts + 1)
            for _ in range(num_days)
        ]
        shifts = [a for _, a in bit_sets]

        def smart_reduce(f, items):
            assert len(items) > 1
            return reduce(f, items[1:], items[0])

        is_working_day = [
            smart_reduce(logical_and, day_shifts[:-1])
            for day_shifts in shifts
        ]

        zero = model.constant(0)

        # Constraints that enforce max shifts in a row
        for day in range(num_days - max_shifts_in_a_row):
            working_more_than_max_shifts = smart_reduce(
                logical_and,
                is_working_day[day:day + max_shifts_in_a_row + 1]
            )
            for nurse in range(num_nurses):
                model.add_constraint(working_more_than_max_shifts[nurse] == zero)

        # Constraints that enforce min two shifts in a row
        for day in range(num_days - 3):
            not_working_min_shifts = logical_and(
                logical_and(shifts[day][-1], is_working_day[day + 1]),
                shifts[day + 2][-1]
            )
            for nurse in range(num_nurses):
                model.add_constraint(not_working_min_shifts[nurse] == zero)

        # Total minutes worked
        total_minutes_worked = None
        for shift in range(num_shifts):
            total_shifts_worked = smart_reduce(
                lambda x, y: x + y,
                [shifts[day][shift] for day in range(num_days)]
            )
            minutes_worked = total_shifts_worked * minutes_per_shift[shift]
            total_minutes_worked = total_minutes_worked + minutes_worked if total_minutes_worked else minutes_worked

        min_minutes = model.constant(min_minutes_per_week)
        max_minutes = model.constant(max_minutes_per_week)
        for nurse in range(num_nurses):
            model.add_constraint(total_shifts_worked[nurse] >= min_minutes)
            model.add_constraint(total_shifts_worked[nurse] <= max_minutes)

        model.lock()

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())

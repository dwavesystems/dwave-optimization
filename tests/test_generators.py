# Copyright 2024 D-Wave Inc.
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

import numpy as np
import unittest
import warnings

try:
    # dimod should fix its own warnings, but just so we're not too coupled,
    # let's ignore import warnings from dimod.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import dimod
except ImportError:
    dimod_found = False
else:
    dimod_found = True

import dwave.optimization
from dwave.optimization.generators import _require


class Test_require(unittest.TestCase):
    # tests for the private _require function

    def test_infinite(self):
        with self.assertRaisesRegex(ValueError, "infs or NaNs"):
            _require("arg1", float("inf"))

    def test_ndim(self):
        self.assertEqual(_require("a", 0), 0)
        np.testing.assert_array_equal(_require("a", 0), [0])  # size up

        with self.assertRaisesRegex(ValueError, "0D array-like"):
            _require("a", [0], ndim=0)


class TestBinPacking(unittest.TestCase):
    def test_input_validations(self):
        # Non-positive bin capacity
        with self.assertRaises(ValueError):
            dwave.optimization.generators.bin_packing([1, 2, 3], 0)

        # Zero-length weights array
        with self.assertRaises(ValueError):
            dwave.optimization.generators.bin_packing([], 1)

        # Negative value in weights array
        with self.assertRaises(ValueError):
            dwave.optimization.generators.bin_packing([1, -2, 3], 5)

        # Item weight greater than bin capacity
        with self.assertRaises(ValueError):
            dwave.optimization.generators.bin_packing([1, 1, 3], 2)

    def test_basics(self):
        weights = [30, 20, 10, 20]
        bin_capacity = 40

        model = dwave.optimization.generators.bin_packing(weights, bin_capacity)

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), 4)
        self.assertEqual(model.is_locked(), True)

        items = next(model.iter_decisions())
        capacity_constraint = next(model.iter_constraints())
        model.states.resize(2)
        items.set_state(0, [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        items.set_state(1, [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(model.objective.state(0), 2.)
        self.assertEqual(capacity_constraint.state(0), 1.)
        self.assertEqual(model.objective.state(1), 1.)
        self.assertEqual(capacity_constraint.state(1), 0.)

    def test_serialization(self):
        weights = [30, 20, 10, 20]
        bin_capacity = 40

        model = dwave.optimization.generators.bin_packing(weights, bin_capacity)

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())


class TestCapacitatedVehicleRouting(unittest.TestCase):
    def test_input_validations(self):
        x = [1, 2, 3]
        demand = [5, 5, 7]
        demand0 = [0, 5, 12]
        num_vehicles = 2
        capacity = 10
        depot_x_y = [11, 22]

        # Zero-length arrays
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=[],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=x,
                locations_y=x)

        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=[],
                locations_y=[])

        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                distances=[])

        # Negative-value arrays
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                distances=[[1, -1], [2, 8]])

        # Unequal-length arrays
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=[1, 2, 3, 4],
                locations_y=x)

        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                distances=[[1, 6], [2, 8]])

        # Not enough vehicles
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=0,
                vehicle_capacity=capacity,
                locations_x=x,
                locations_y=x)

        # No capacity per vehicle
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=0,
                locations_x=x,
                locations_y=x)

        # Not enough total capacity for total demand
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=[30, 30, 31],
                number_of_vehicles=3,
                vehicle_capacity=30,
                locations_x=x,
                locations_y=x)

        # No customers when depot in demand vector
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=[0],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=[11],
                locations_y=[22])

        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=[0],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                distances=[6])

        # Depot location in both demand vector and argument
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand0,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=x,
                locations_y=x,
                depot_x_y=depot_x_y)

        # Depot location with distances array
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand0,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                distances=[[1, 6, 3], [2, 2, 8], [4, 6, 7]],
                depot_x_y=depot_x_y)

        # Demand vector has zero for depot and customer
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=[0, 44.6, 0],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=x,
                locations_y=x)

        # Demand vector has zero for customer
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=[22, 44.6, 0],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                locations_x=x,
                locations_y=x)

        # Demand vector first element is not zero for distances array
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                distances=[[1, 6, 3], [2, 2, 8], [4, 6, 7]], )

    def test_basics(self):
        num_vehicles = 2

        # Depot at origin from depot_x_y default
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            demand=[5, 5],
            number_of_vehicles=num_vehicles,
            vehicle_capacity=10,
            locations_x=[0, 3],
            locations_y=[2, 0],
            depot_x_y=[0, 0])

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), num_vehicles)
        self.assertEqual(model.num_nodes(), 35)
        self.assertEqual(model.num_edges(), 46)
        self.assertEqual(model.is_locked(), True)

        model.states.resize(1)
        route = next(model.iter_decisions())
        self.assertEqual(route.num_disjoint_lists(), 2)
        route.set_state(0, [[0], [1]])
        self.assertEqual(model.objective.state(0), 10)

        # Depot at (0, 1.5) from demand[0] == 0
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            demand=[0, 5],
            number_of_vehicles=num_vehicles,
            vehicle_capacity=10,
            locations_x=[0, 3],
            locations_y=[1.5, 1.5])

        model.states.resize(1)
        route = next(model.iter_decisions())
        self.assertEqual(route.num_disjoint_lists(), 2)
        route.set_state(0, [[0], []])
        self.assertEqual(model.objective.state(0), 6)

        # Depot at (3, 2) from depot_x_y
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            demand=[5, 5],
            number_of_vehicles=num_vehicles,
            vehicle_capacity=10,
            locations_x=[3, 6],
            locations_y=[4, 2],
            depot_x_y=[3, 2])

        model.states.resize(1)
        route = next(model.iter_decisions())
        self.assertEqual(route.num_disjoint_lists(), 2)
        route.set_state(0, [[0], [1]])
        self.assertEqual(model.objective.state(0), 10)

        # Test asymmetric distances
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            demand=[0, 10, 10],
            number_of_vehicles=num_vehicles,
            vehicle_capacity=10,
            distances=[[0, 1, 3], [2, 0, np.sqrt(17)], [4, np.sqrt(13), 0]])

        model.states.resize(1)
        route = next(model.iter_decisions())
        self.assertEqual(route.num_disjoint_lists(), 2)
        route.set_state(0, [[0], [1]])
        self.assertEqual(model.objective.state(0), 10)

    def test_cvrplib_P_n19_k2(self):
        # http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n19-k2.vrp
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            locations_x=[30, 37, 49, 52, 31, 52, 42, 52, 57, 62, 42, 27, 43, 58, 37, 61, 62, 63, 45],
            locations_y=[40, 52, 43, 64, 62, 33, 41, 41, 58, 42, 57, 68, 67, 27, 69, 33, 63, 69, 35],
            demand=[0, 19, 30, 16, 23, 11, 31, 15, 28, 14, 8, 7, 14, 19, 11, 26, 17, 6, 15],
            number_of_vehicles=2,
            vehicle_capacity=160)

        model.states.resize(1)
        route = next(model.iter_decisions())
        route.set_state(0, [[i - 1 for i in [4, 11, 14, 12, 3, 17, 16, 8, 6]],
                            [i - 1 for i in [18, 5, 13, 15, 9, 7, 2, 10, 1]]])
        self.assertGreater(model.objective.state(0), 212)
        self.assertLess(model.objective.state(0), 213)

    def test_serialization(self):
        # http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n19-k2.vrp
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            locations_x=[30, 37, 49, 52, 31, 52, 42, 52, 57, 62, 42, 27, 43, 58, 37, 61, 62, 63, 45],
            locations_y=[40, 52, 43, 64, 62, 33, 41, 41, 58, 42, 57, 68, 67, 27, 69, 33, 63, 69, 35],
            demand=[0, 19, 30, 16, 23, 11, 31, 15, 28, 14, 8, 7, 14, 19, 11, 26, 17, 6, 15],
            number_of_vehicles=2,
            vehicle_capacity=160)

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())

    def test_state_serialization(self):
        # http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n19-k2.vrp
        model = dwave.optimization.generators.capacitated_vehicle_routing(
            locations_x=[30, 37, 49, 52, 31, 52, 42, 52, 57, 62, 42, 27, 43, 58, 37, 61, 62, 63, 45],
            locations_y=[40, 52, 43, 64, 62, 33, 41, 41, 58, 42, 57, 68, 67, 27, 69, 33, 63, 69, 35],
            demand=[0, 19, 30, 16, 23, 11, 31, 15, 28, 14, 8, 7, 14, 19, 11, 26, 17, 6, 15],
            number_of_vehicles=2,
            vehicle_capacity=160)

        model.states.resize(1)

        routes, = model.iter_decisions()

        routes.set_state(0, [[i - 1 for i in [4, 11, 14, 12, 3, 17, 16, 8, 6]],
                             [i - 1 for i in [18, 5, 13, 15, 9, 7, 2, 10, 1]]])

        # just smoke test
        with model.states.to_file() as f:
            model.states.from_file(f)
        self.assertGreater(model.objective.state(0), 212)
        self.assertLess(model.objective.state(0), 213)


class TestCapacitatedVehicleRoutingTimeWindow(unittest.TestCase):
    def test_input_validations(self):
        time_distances = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        demand = [5, 5, 7]
        demand0 = [0, 5, 12]
        num_vehicles = 2
        capacity = 10
        time_window_open = [0, 0]
        time_window_close = [100, 100]
        service_time = [1, 1]

        # Zero-length arrays
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=[],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time
            )

        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=demand0,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=[],
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Negative-value arrays
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=demand0,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=[[1, 2, 3], [1, 2, 3], [1, 2, -3]],
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Unequal-length arrays
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():  # older version of NumPy might throw deprecation
                warnings.simplefilter("ignore")
                dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                    demand=demand0,
                    number_of_vehicles=num_vehicles,
                    vehicle_capacity=capacity,
                    time_distances=[[1, 2, 3], [1, 2], [1, 2, 3]],
                    time_window_open=time_window_open,
                    time_window_close=time_window_close,
                    service_time=service_time)

        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=demand0,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=[[1, 6], [2, 8]],
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Not enough vehicles
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=demand0,
                number_of_vehicles=0,
                vehicle_capacity=capacity,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # No capacity per vehicle
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=demand0,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=0,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Not enough total capacity for total demand
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=[30, 30, 31],
                number_of_vehicles=3,
                vehicle_capacity=30,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Non-zero depot demand
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=demand,
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Demand vector has zero for depot and customer
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=[0, 44.6, 0],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

        # Demand vector has zero for customer
        with self.assertRaises(ValueError):
            dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
                demand=[22, 44.6, 0],
                number_of_vehicles=num_vehicles,
                vehicle_capacity=capacity,
                time_distances=time_distances,
                time_window_open=time_window_open,
                time_window_close=time_window_close,
                service_time=service_time)

    def test_basics(self):
        num_vehicles = 2

        model = dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
            demand=[0, 5, 5],
            number_of_vehicles=num_vehicles,
            vehicle_capacity=10,
            time_distances=[[1, 6, 3], [2, 2, 8], [4, 6, 7]],
            time_window_open=[0, 0, 0],
            time_window_close=[20, 20, 20],
            service_time=[0, 0, 0])

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), 12)
        self.assertEqual(model.is_locked(), True)

        model.states.resize(1)
        route = next(model.iter_decisions())
        self.assertEqual(route.num_disjoint_lists(), 2)
        route.set_state(0, [[0], [1]])
        self.assertEqual(model.objective.state(0), 15)

        # Test asymmetric distances
        model = dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
            demand=[0, 10, 10],
            number_of_vehicles=num_vehicles,
            vehicle_capacity=10,
            time_distances=[[0, 1, 3], [2, 0, np.sqrt(17)], [4, np.sqrt(13), 0]],
            time_window_open=[0, 0, 0],
            time_window_close=[0, 100, 100],
            service_time=[0, 0, 0]
        )

        model.states.resize(1)
        route = next(model.iter_decisions())
        self.assertEqual(route.num_disjoint_lists(), 2)
        route.set_state(0, [[0], [1]])
        self.assertEqual(model.objective.state(0), 10)

    def test_serialization(self):
        model = dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
            time_distances=[[0, 14, 19, 32],
                            [14, 0, 15, 19],
                            [19, 15, 0, 21],
                            [32, 19, 21, 0]],
            demand=[0, 19, 30, 16],
            number_of_vehicles=2,
            vehicle_capacity=40,
            time_window_open=[0, 0, 0, 0],
            time_window_close=[0, 100, 100, 100],
            service_time=[0, 0, 0, 0]
        )

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())

    def test_state_serialization(self):
        model = dwave.optimization.generators.capacitated_vehicle_routing_with_time_windows(
            time_distances=[[0, 14, 19, 32],
                            [14, 0, 15, 19],
                            [19, 15, 0, 21],
                            [32, 19, 21, 0]],
            demand=[0, 19, 30, 16],
            number_of_vehicles=2,
            vehicle_capacity=40,
            time_window_open=[0, 0, 0, 0],
            time_window_close=[0, 100, 100, 100],
            service_time=[0, 2, 2, 2]
        )

        model.states.resize(1)

        routes, = model.iter_decisions()

        routes.set_state(0, [[i-1 for i in [1, 2]],
                             [i-1 for i in [3]]])

        # just smoke test
        with model.states.to_file() as f:
            model.states.from_file(f)
        self.assertGreater(model.objective.state(0), 110)
        self.assertLess(model.objective.state(0), 115)


@unittest.skipUnless(dimod_found, "No dimod installed")
class TestConstrainedQuadraticModel(unittest.TestCase):
    def test_empty(self):
        cqm = dimod.ConstrainedQuadraticModel()
        model = dwave.optimization.generators._from_constrained_quadratic_model(cqm)
        self.assertEqual(model.num_symbols(), 0)


class TestFlowShopScheduling(unittest.TestCase):
    def test_basics(self):
        processing_times = [[10, 5, 7], [20, 10, 15]]

        model = dwave.optimization.generators.flow_shop_scheduling(processing_times=processing_times)

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), 0)
        self.assertEqual(model.num_nodes(), 27)
        self.assertEqual(model.num_edges(), 38)
        self.assertEqual(model.is_locked(), True)

        model.states.resize(1)
        order = next(model.iter_decisions())
        self.assertEqual(order.size(), 3)
        order.set_state(0, [0, 1, 2])
        self.assertEqual(model.objective.state(0), 55)

    def test_non_integer_processing_times(self):
        processing_times = np.asarray([[10, 5, 7], [20, 10, 15]], dtype=float) / 2

        model = dwave.optimization.generators.flow_shop_scheduling(processing_times=processing_times)

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), 0)
        self.assertEqual(model.num_nodes(), 27)
        self.assertEqual(model.num_edges(), 38)
        self.assertEqual(model.is_locked(), True)

        model.states.resize(1)
        order = next(model.iter_decisions())
        self.assertEqual(order.size(), 3)
        order.set_state(0, [0, 1, 2])
        self.assertAlmostEqual(model.objective.state(0), 55 / 2)

    def test_serialization(self):
        processing_times = [[10, 5, 7], [20, 10, 15]]
        model = dwave.optimization.generators.flow_shop_scheduling(processing_times=processing_times)

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())


class TestJobShopScheduling(unittest.TestCase):
    def test_1indexed(self):
        times = [[2, 1, 3],
                 [4, 1, 2],
                 [1, 1, 2]]
        machines = [[1, 2, 3],
                    [3, 1, 2],
                    [3, 2, 1]]

        model = dwave.optimization.generators.job_shop_scheduling(times, machines)

        # check that we have the decisions we expect
        times, = model.iter_decisions()

        model.states.resize(1)

        times.set_state(0, [[0, 2, 4], [2, 6, 0], [6, 4, 2]])

        # objective is exactly the makespan and the model is feasible
        self.assertEqual(model.objective.state(0), 7)
        for constraint in model.iter_constraints():
            self.assertTrue(constraint.state(0))

    def test_1x4(self):
        times = [1, 2, 1, 1]
        machines = [1, 2, 3, 4]  # 1-indexed

        model = dwave.optimization.generators.job_shop_scheduling(times, machines)

        # check that we have the decisions we expect
        times, = model.iter_decisions()

        model.states.resize(1)

        with self.subTest("feasible and optimal"):
            # a
            #  aa
            #    a
            #     a

            times.set_state(0, [[0, 1, 3, 4]])

            # objective is exactly the makespan and the model is feasible
            self.assertEqual(model.objective.state(0), 5)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

        with self.subTest("feasible and not optimal"):
            # a
            #  aa
            #     a
            #      a

            times.set_state(0, [[0, 1, 4, 5]])

            self.assertEqual(model.objective.state(0), 6)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

        with self.subTest("feasible and not optimal"):
            # a
            #  aa
            #   a
            #    a

            times.set_state(0, [[0, 1, 2, 3]])

            self.assertEqual(model.objective.state(0), 4)
            self.assertFalse(all(c.state(0) for c in model.iter_constraints()))

    def test_3x1(self):
        times = [[2, 1],
                 [1, 2],
                 [1, 1]]
        machines = [[0, 1],
                    [1, 0],
                    [0, 1]]

        model = dwave.optimization.generators.job_shop_scheduling(times, machines)

        # check that we have the decisions we expect
        times, = model.iter_decisions()

        model.states.resize(1)

        # for the following tests, we'll call the jobs a,b,c

        with self.subTest("feasible and optimal"):
            # aacb
            # bbac

            times.set_state(0, [[0, 2], [3, 0], [2, 3]])

            # objective is exactly the makespan and the model is feasible
            self.assertEqual(model.objective.state(0), 4)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

        with self.subTest("feasible and not optimal"):
            # aabc
            # bba c

            times.set_state(0, [[0, 2], [2, 0], [3, 4]])

            self.assertEqual(model.objective.state(0), 5)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

    def test_3x3(self):
        times = [[2, 1, 3],
                 [4, 1, 2],
                 [1, 1, 2]]
        machines = [[0, 1, 2],
                    [2, 0, 1],
                    [2, 1, 0]]

        model = dwave.optimization.generators.job_shop_scheduling(times, machines)

        # check that we have the decisions we expect
        times, = model.iter_decisions()

        model.states.resize(1)

        # for the following tests, we'll call the jobs a,b,c

        with self.subTest("feasible and optimal"):
            # aabbbbc
            #   a c b
            # bbccaaa

            times.set_state(0, [[0, 2, 4], [2, 6, 0], [6, 4, 2]])

            # objective is exactly the makespan and the model is feasible
            self.assertEqual(model.objective.state(0), 7)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

        with self.subTest("feasible and not optimal"):
            # aabbbbc
            #   a c  b
            # bbccaaa

            times.set_state(0, [[0, 2, 4], [2, 7, 0], [6, 4, 2]])

            # objective is exactly the makespan and the model is feasible
            self.assertEqual(model.objective.state(0), 8)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

        with self.subTest("infeasible - overlapping job"):
            # aabbbbc
            #  a  c b
            # bbccaaa

            times.set_state(0, [[0, 1, 4], [2, 6, 0], [6, 4, 2]])

            # objective is exactly the makespan and the model is infeasible
            self.assertEqual(model.objective.state(0), 7)
            self.assertFalse(all(c.state(0) for c in model.iter_constraints()))

        with self.subTest("infeasible - overlapping machine"):
            # aabbbbc
            #   a c b
            # b.c aaa

            times.set_state(0, [[0, 2, 4], [2, 6, 0], [6, 4, 1]])

            # objective is exactly the makespan and the model is infeasible
            self.assertEqual(model.objective.state(0), 7)
            self.assertFalse(all(c.state(0) for c in model.iter_constraints()))

    def test_4x1(self):
        times = np.asarray([[1, 2, 1, 1]]).T
        machines = np.asarray([[1, 1, 1, 1]]).T

        model = dwave.optimization.generators.job_shop_scheduling(times, machines)

        # check that we have the decisions we expect
        times, = model.iter_decisions()

        model.states.resize(1)

        with self.subTest("feasible and optimal"):
            # a
            #  bb
            #    c
            #     d

            times.set_state(0, [[0], [1], [3], [4]])

            # objective is exactly the makespan and the model is feasible
            self.assertEqual(model.objective.state(0), 5)
            for constraint in model.iter_constraints():
                self.assertTrue(constraint.state(0))

    def test_input_checking(self):
        # empty
        with self.assertRaises(ValueError):
            dwave.optimization.generators.job_shop_scheduling([], [])

        # different shape
        with self.assertRaises(ValueError):
            dwave.optimization.generators.job_shop_scheduling([[0]], [[0, 1]])

        # 3d
        with self.assertRaises(ValueError):
            dwave.optimization.generators.job_shop_scheduling([[[0]]], [[[0]]])

        # not permutation
        times = [[2, 1, 3],
                 [4, 1, 2],
                 [1, 1, 2]]
        machines = [[0, 1, 1],
                    [2, 0, 1],
                    [2, 1, 0]]
        with self.assertRaises(ValueError):
            dwave.optimization.generators.job_shop_scheduling(times, machines)

    def test_serialization(self):
        times = [[2, 1, 3],
                 [4, 1, 2],
                 [1, 1, 2]]
        machines = [[0, 1, 2],
                    [2, 0, 1],
                    [2, 1, 0]]

        model = dwave.optimization.generators.job_shop_scheduling(times, machines)

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())


class TestKnapsack(unittest.TestCase):
    def test_basics(self):
        weights = [30, 10, 40, 20]
        values = [10, 20, 30, 40]
        capacity = 20

        model = dwave.optimization.generators.knapsack(values, weights, capacity)

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), 1)
        self.assertEqual(model.is_locked(), True)

        items = next(model.iter_decisions())
        capacity_constraint = next(model.iter_constraints())
        model.states.resize(2)
        items.set_state(0, [3])
        items.set_state(1, [0, 1, 2])
        self.assertEqual(model.objective.state(0), -40.)
        self.assertEqual(capacity_constraint.state(0), 1.)
        self.assertEqual(model.objective.state(1), -60.)
        self.assertEqual(capacity_constraint.state(1), 0.)

    def test_serialization(self):
        weights = [30, 10, 40, 20]
        values = [10, 20, 30, 40]
        capacity = 20

        model = dwave.optimization.generators.knapsack(values, weights, capacity)

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())


class TestQuadraticAssignment(unittest.TestCase):
    def test_input_validations(self):
        distance_matrix0 = [[0, 5, 3],
                            [5, 0, 2],
                            [3, 2, 0]]
        flow_matrix0 = [[0, 1, 3],
                        [4, 0, 2],
                        [1, 1, 0]]

        distance_matrix1 = [[0, 5, 3],
                            [-5, 0, 2],
                            [3, 2, 0]]
        flow_matrix1 = [[0, 1, 3],
                        [4, 0, -2],
                        [1, 1, 0]]

        distance_matrix2 = [[0, 5, 3],
                            [5, 0, 2],
                            [3, 2]]
        flow_matrix2 = [[0, 1, 3]]

        # Zero-length matrix
        with self.assertRaises(ValueError):
            dwave.optimization.generators.quadratic_assignment([], [])

        with self.assertRaises(ValueError):
            dwave.optimization.generators.quadratic_assignment([], flow_matrix0)

        with self.assertRaises(ValueError):
            dwave.optimization.generators.quadratic_assignment(distance_matrix0, [])

        # Negative-value in the matrix
        with self.assertRaises(ValueError):
            dwave.optimization.generators.quadratic_assignment(distance_matrix1, flow_matrix0)

        with self.assertRaises(ValueError):
            dwave.optimization.generators.quadratic_assignment(distance_matrix0, flow_matrix1)

        # Unacceptable matrix shape
        with self.assertRaises(ValueError):
            dwave.optimization.generators.quadratic_assignment(distance_matrix0, flow_matrix2)

        # NumPy raises an error for this, but in some version there are deprecation warnings so
        # disable this test for now in order to keep the tests output warning-free
        # with self.assertRaises(ValueError):
        #     dwave.optimization.generators.quadratic_assignment(distance_matrix2, flow_matrix0)

    def test_basics(self):
        distance_matrix = [[0, 5, 3],
                           [5, 0, 2],
                           [3, 2, 0]]
        flow_matrix = [[0, 1, 3],
                       [4, 0, 2],
                       [1, 1, 0]]

        model = dwave.optimization.generators.quadratic_assignment(distance_matrix, flow_matrix)

        self.assertEqual(model.num_decisions(), 1)
        self.assertEqual(model.num_constraints(), 0)
        self.assertEqual(model.num_nodes(), 7)
        self.assertEqual(model.num_edges(), 7)
        self.assertEqual(model.is_locked(), True)

        model.states.resize(1)
        order = next(model.iter_decisions())
        self.assertEqual(order.size(), 3)
        order.set_state(0, [0, 1, 2])
        self.assertEqual(model.objective.state(0), 43)

    def test_serialization(self):
        distance_matrix = [[0, 5, 3],
                           [5, 0, 2],
                           [3, 2, 0]]
        flow_matrix = [[0, 1, 3],
                       [4, 0, 2],
                       [1, 1, 0]]

        model = dwave.optimization.generators.quadratic_assignment(distance_matrix, flow_matrix)

        with model.to_file() as f:
            copy = dwave.optimization.Model.from_file(f)

        # a few smoke test checks
        self.assertEqual(model.num_symbols(), copy.num_symbols())
        self.assertEqual(model.state_size(), copy.state_size())


class TestTravelingSalesperson(unittest.TestCase):
    def test_asymmetric(self):
        # going 0->1->2 is distance 9, but 2->1->0 is distance 3
        D = [[0, 3, 1], [1, 0, 3], [3, 1, 0]]

        tsp = dwave.optimization.generators.traveling_salesperson(D)
        route, = tsp.iter_decisions()

        tsp.states.resize(2)

        route.set_state(0, [0, 1, 2])
        route.set_state(1, [2, 1, 0])

        self.assertEqual(tsp.objective.state(0), 9)
        self.assertEqual(tsp.objective.state(1), 3)

    def test_exceptions(self):
        with self.subTest("rectangular distance matrix"):
            D = [[0, 3, 1], [1, 0, 3]]
            with self.assertRaisesRegex(ValueError, "square"):
                dwave.optimization.generators.traveling_salesperson(D)

    def test_nonzero_diagonal(self):
        D = [[1, 3, 1], [1, 2, 3], [3, 1, 3]]

        tsp = dwave.optimization.generators.traveling_salesperson(D)
        route, = tsp.iter_decisions()

        tsp.states.resize(1)

        route.set_state(0, [0, 1, 2])

        self.assertEqual(tsp.objective.state(0), 9)  # diagonal doesn't contribute

    def test_scalar(self):
        # this is silly, but valid
        tsp = dwave.optimization.generators.traveling_salesperson(0)
        route, = tsp.iter_decisions()

        tsp.states.resize(1)

        route.set_state(0, [0])
        self.assertEqual(tsp.objective.state(0), 0)

    def test_serialization(self):
        D = [[0, 1, 2, 3, 4],
             [1, 0, 5, 6, 7],
             [2, 5, 0, 8, 9],
             [3, 6, 8, 0, 10],
             [4, 7, 9, 10, 0]]

        tsp = dwave.optimization.generators.traveling_salesperson(D)

        with tsp.to_file() as f:
            recreated_model = tsp.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), tsp.num_nodes())

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

"""Model generators for optimization problems."""

from __future__ import annotations

import itertools
import typing

import numpy as np
import numpy.typing

from dwave.optimization.mathematical import add, logical_or, maximum, where
from dwave.optimization.model import Model

__all__ = [
    "bin_packing",
    "capacitated_vehicle_routing",
    "capacitated_vehicle_routing_with_time_windows",
    "flow_shop_scheduling",
    "job_shop_scheduling",
    "knapsack",
    "quadratic_assignment",
    "traveling_salesperson",
]


def _require(argname: str,
             array_like: numpy.typing.ArrayLike,
             *,
             dtype=None,
             ndim: typing.Optional[int] = None,
             nonnegative: bool = False,
             positive: bool = False,
             square: bool = False
             ) -> np.ndarray:
    """Coerce the given array-like into the form we want and raise a consistent
    error message if it cannot be coerced.
    """
    try:
        array = np.asarray(array_like, order="C", dtype=dtype)
    except (ValueError, TypeError) as err:
        raise ValueError(f"`{argname}` must be an array-like of numbers") from err

    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"`{argname}` must be an array-like of numbers")

    # Disallow inf and NaN. We could make this toggleable with a kwarg if there
    # is eventually a reason for it
    try:
        array = np.asarray_chkfinite(array)
    except ValueError as err:
        raise ValueError(f"'{argname}' must not contain infs or NaNs") from err

    if ndim is not None and array.ndim != ndim:
        # we don't "size down"
        if array.ndim > ndim:
            raise ValueError(f"'{argname}' must be a {ndim}D array-like")

        # We do "size up", following NumPy conventions that are usually permissive.
        # e.g. it's silly, but well defined to have a distance matrix with only
        # a single location. Or to have a JSS instance with only one job.
        while array.ndim < ndim:
            array = array[np.newaxis]

    if positive and (array <= 0).any():
        raise ValueError(f"`{argname}` must be positive")

    if nonnegative and (array < 0).any():
        raise ValueError(f"`{argname}` must be nonnegative")

    if square and len(set(array.shape)) > 1:
        raise ValueError(f"`{argname}` must be a square array-like")

    return array


# Dev note: this is currently private as it's not optimized and doesn't do the full
# set of features we'd eventually like
def _from_constrained_quadratic_model(cqm) -> Model:
    """Construct a NL model from a :class:`dimod.ConstrainedQuadraticModel`."""

    for v in cqm.variables:
        if cqm.vartype(v).name != "BINARY":
            raise ValueError("CQM must only have binary variables")

    model = Model()

    if not cqm.num_variables():
        return model

    x = model.binary(cqm.num_variables())

    def quadratic_model(qm):
        # Get the indices of the variables in the QM
        # In the future we could test if this is a range and do basic indexing in that case
        indices = model.constant([cqm.variables.index(v) for v in qm.variables])

        # Get the biases in COO format
        # We should add `to_numpy_vectors()` to dimod objective/constraints
        linear = model.constant([qm.get_linear(v) for v in qm.variables])

        if qm.is_linear():
            out = (linear * x[indices]).sum()
        else:
            irow = []
            icol = []
            quadratic = []
            for u, v, bias in qm.iter_quadratic():
                irow.append(qm.variables.index(u))
                icol.append(qm.variables.index(v))
                quadratic.append(bias)

            out = model.quadratic_model(x[indices], (np.asarray(quadratic), (irow, icol)), linear)

        if qm.offset:
            return out + model.constant(qm.offset)

        return out

    # Minimize the objective
    model.minimize(quadratic_model(cqm.objective))

    for comp in cqm.constraints.values():
        lhs = quadratic_model(comp.lhs)
        rhs = model.constant(comp.rhs)

        if comp.sense.value == "==":
            model.add_constraint(lhs == rhs)
        elif comp.sense.value == "<=":
            model.add_constraint(lhs <= rhs)
        elif comp.sense.value == ">=":
            model.add_constraint(rhs <= lhs)
        else:
            raise RuntimeError("unexpected sense")

    return model


def bin_packing(weights: numpy.typing.ArrayLike,
                capacity: float
                ) -> Model:
    r"""Generate a model encoding a bin packing problem.

    The bin packing problem,
    `BPP <https://en.wikipedia.org/wiki/Bin_packing_problem>`_,
    seeks to find the smallest number of bins that will fit
    a set of weighted items given that each bin has a weight capacity.

    Args:
        weights:
            A 1D |array-like|_ (row vector or Python list) of weights per item.
            Weights can be any non-negative number.

        capacity:
            Maximum capacity of each bin. Must be a positive number.

    Returns:
        A model encoding the BPP problem.
    """

    weights = _require("weights", weights, dtype=float, nonnegative=True, ndim=1)

    if weights.shape[0] == 0:
        raise ValueError("the number of items must be positive")

    if capacity <= 0:
        raise ValueError("`capacity` must be a positive number")

    if any(weight > capacity for weight in weights):
        raise ValueError("each weight in `weights` must be smaller than bin_capacity")

    num_items = weights.shape[0]

    # Construct model
    model = Model()

    # Add constants
    weights = model.constant(weights)
    capacity = model.constant(capacity)

    # Create disjoint bit sets to represent bins
    max_bins = num_items
    main, bins = model.disjoint_bit_sets(num_items, max_bins)

    # Ensure that the weight for each bin does not exceed capacity
    for i in range(max_bins):
        bin_array = bins[i]
        model.add_constraint((bin_array * weights).sum() <= capacity)

    # Minimize the number of bins that are non-empty
    constant_one = model.constant(1)
    obj_val = model.constant(0)
    for i in range(max_bins):
        obj_val += (bins[i].sum() >= constant_one)

    model.minimize(obj_val)

    model.lock()
    return model


def capacitated_vehicle_routing(demand: numpy.typing.ArrayLike,
                                number_of_vehicles: int,
                                vehicle_capacity: float,
                                distances: typing.Optional[numpy.typing.ArrayLike] = None,
                                locations_x: typing.Optional[numpy.typing.ArrayLike] = None,
                                locations_y: typing.Optional[numpy.typing.ArrayLike] = None,
                                depot_x_y: typing.Optional[numpy.typing.ArrayLike] = None
                                ) -> Model:
    r"""Generate a model encoding a capacitated vehicle routing problem.

    The capacitated vehicle routing problem,
    `CVRP <https://en.wikipedia.org/wiki/Vehicle_routing_problem>`_,
    is to find the shortest possible routes for a fleet of vehicles delivering
    to multiple customer locations from a central depot. Vehicles have a
    specified delivery capacity, and on the routes to locations and then back
    to the depot, no vehicle is allowed to exceed its carrying capacity.

    Args:
        demand:
            Customer demand, as an |array-like|_. If ``distances`` is specified,
            the first element must be zero. If ``distances`` is not specified
            and the first element is zero, ``[locations_x[0], locations_y[0]]``
            must be the location of the depot. Elements other than the first
            must be positive numbers.
        number_of_vehicles:
            Number of available vehicles, as a positive integer.
        vehicle_capacity:
            Maximum capacity for any vehicle. The total delivered demand by any
            vehicle on any route must not exceed this value.
        distances:
            Distances between **all** the problem's locations, as an |array-like|_
            of positive numbers, including both customer sites and the depot.
            When specified, the first element of ``demand`` must be zero and
            specifying ``X coordinates`` or ``X coordinates`` is not supported
        locations_x:
            X coordinates, as an |array-like|_, of locations for customers, and
            optionally the depot. When specified, 2D Euclidean distances are
            calculated and specifying ``distances`` is not supported. If the
            first element represents the X coordinate of the depot, the first
            element of ``demand`` must be zero.
        locations_y:
            Y coordinates, as an |array-like|_, of locations for customers, and
            optionally the depot. When specified, 2D Euclidean distances are
            calculated and specifying ``distances`` is not supported. If the
            first element represents the Y coordinate of the depot, the first
            element of ``demand`` must be zero.
        depot_x_y:
            Location of the depot, as an |array-like|_ of exactly two elements,
            ``[X, Y]``. Required if the first element of ``demand`` is nonzero
            and ``distances`` is not specified; not allowed otherwise.

    Returns:
        A model encoding the CVRP problem.

    Notes:
        The model uses a :class:`~dwave.optimization.model.Model.disjoint_lists`
        class as the decision variable being optimized, with permutations of its
        sublist representing various itineraries for each vehicle.
    """

    if not isinstance(number_of_vehicles, int):
        raise ValueError("`number_of_vehicles` must be an integer.")

    if number_of_vehicles < 1:
        raise ValueError("`number_of_vehicles` must be at least 1."
                         f" Got {number_of_vehicles}.")

    if vehicle_capacity <= 0:
        raise ValueError("`vehicle_capacity` must be a positive number."
                         f" Got {vehicle_capacity}.")

    if distances is not None and (locations_x is not None or locations_y is not None):
        raise ValueError("Either `locations_x` and `locations_y` or `distances`"
                         f" can be specified. Got both input formats.")

    demand = _require("demand", demand, dtype=float, ndim=1, nonnegative=True)

    if demand.sum() > number_of_vehicles * vehicle_capacity:
        raise ValueError(f"Total demand, {demand.sum()}, exceeds the "
                         f"total capacity, {number_of_vehicles * vehicle_capacity}.")

    if distances is not None:

        if depot_x_y is not None:
            raise ValueError("`depot_x_y` and `distances` cannot be specified together.")

        distances_array = _require("distances", distances, ndim=2, nonnegative=True, square=True)

        if distances_array.shape[0] < 2:
            raise ValueError("Number of rows in `distances` must be at least 2."
                             f" Got {distances_array.shape[0]} rows.")

        if distances_array.shape[0] != len(demand):
            raise ValueError("Number of rows in `distances` and length of `demand` must be equal."
                             f" Got {distances_array.shape[0]} and"
                             f" {len(demand)}, respectively.")

        if demand[0] != 0:
            raise ValueError("`demand[0]` must be zero when `distances` is specified.")

        customer_demand = demand[1:]
        distance_matrix = distances_array[1:, 1:]
        depot_distance_vector = distances_array[0][1:]
        depot_distance_vector_return = distances_array[1:, 0]

    else:
        x = _require("locations_x", locations_x, ndim=1, dtype=float)
        y = _require("locations_x", locations_y, ndim=1, dtype=float)

        if not len(x) == len(y) == len(demand):
            raise ValueError("Lengths of `locations_x`, `locations_y`, and `demand`"
                             f" must be equal. Got lengths {len(x)}, {len(y)}, and"
                             f" {len(demand)}, respectively.")
        if len(x) < 1:
            raise ValueError("Lengths of `locations_x`, `locations_y`, and `demand`"
                             " must be at least 1. Got length zero.")

        if demand[0] == 0:
            if len(x) < 2:
                raise ValueError("Lengths of `locations_x` and `locations_y` must"
                                 f" be at least 2 when `demand[0]=0`.")

            customer_demand = demand[1:]
            customer_locations_x = locations_x[1:]
            customer_locations_y = locations_y[1:]

            if depot_x_y is not None:
                raise ValueError("`depot_x_y` cannot be provided when "
                                 "`demand[0]` is zero.")

            depot_x_y = np.asarray([locations_x[0], locations_y[0]])

        else:

            if depot_x_y is None:
                raise ValueError("`depot_x_y` must be provided when `demand[0]` is not"
                                 " 0 and `locations_x` and `locations_y` are specified.")
            customer_demand = demand
            customer_locations_x = locations_x
            customer_locations_y = locations_y
            depot_x_y = np.asarray([depot_x_y[0], depot_x_y[1]])

        customer_locations_x = customer_locations_x - depot_x_y[0]
        customer_locations_y = customer_locations_y - depot_x_y[1]

        x0, x1 = np.meshgrid(customer_locations_x, customer_locations_x)
        y0, y1 = np.meshgrid(customer_locations_y, customer_locations_y)
        distance_matrix = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        depot_distance_vector = depot_distance_vector_return = np.sqrt(
            customer_locations_x ** 2 + customer_locations_y ** 2)

    if (customer_demand == 0).any():
        raise ValueError("Only the first element of `demand` can be zero."
                         f" Got zeros for indices {list(np.where(demand == 0)[0])}.")

    num_customers = len(customer_demand)

    # Construct the model
    model = Model()

    # Add the constants
    customer_dist = model.constant(distance_matrix)
    depot_dist = model.constant(depot_distance_vector)
    depot_dist_return = model.constant(depot_distance_vector_return)
    demand = model.constant(customer_demand)
    capacity = model.constant(vehicle_capacity)

    # Add the decision variable
    routes_decision, routes = model.disjoint_lists(
        primary_set_size=num_customers,
        num_disjoint_lists=number_of_vehicles)

    # The objective is to minimize the distance traveled.
    # This is calculated by adding the distance from the depot to the 1st customer
    # to the distance from the last customer back to the depot, using indices on
    # `depot_dist` and `depot_dist_return`, respectively, selected by the choice of
    # decision variable's routes, and to the distances between customers per route,
    # using indices on `customer_dist` selected similarly.
    # Python represents this slicing of a list with slice indices ``[:-1]`` and ``[1:]``,
    # respectively, because Python slices are defined in the format of a
    # mathematical interval (see https://en.wikipedia.org/wiki/Interval_(mathematics))
    # [a, b) and list indices start at zero. For example, ``routes[r][:-1]``
    # excludes the last element of the sliced list.
    route_costs = []
    for r in range(number_of_vehicles):
        route_costs.append(depot_dist[routes[r][:1]].sum())
        route_costs.append(depot_dist_return[routes[r][-1:]].sum())
        route_costs.append(customer_dist[routes[r][:-1], routes[r][1:]].sum())

        model.add_constraint(demand[routes[r]].sum() <= capacity)

    model.minimize(add(*route_costs))

    model.lock()
    return model


def capacitated_vehicle_routing_with_time_windows(demand: numpy.typing.ArrayLike,
                                                  number_of_vehicles: int,
                                                  vehicle_capacity: float,
                                                  time_distances: numpy.typing.ArrayLike,
                                                  time_window_open: numpy.typing.ArrayLike,
                                                  time_window_close: numpy.typing.ArrayLike,
                                                  service_time: numpy.typing.ArrayLike
                                                  ) -> Model:
    r"""Generate a model encoding a capacitated vehicle routing problem with time windows.

    The capacitated vehicle routing problem with time windows,
    `CVRPTW <https://en.wikipedia.org/wiki/Vehicle_routing_problem>`_,
    is to find the shortest possible routes for a fleet of vehicles delivering
    to multiple customer locations from a central depot. Each customer should
    be served after time_window_open and before time_window_close. Vehicles have a
    specified delivery capacity, and on the routes to locations and then back
    to the depot, no vehicle is allowed to exceed its carrying capacity.

    Args:
        demand:
            Customer demand, as an |array-like|_. The first element is the depot and must
            be zero.
        number_of_vehicles:
            Number of available vehicles, as a positive integer.
        vehicle_capacity:
            Maximum capacity for any vehicle. The total delivered demand by any
            vehicle on any route must not exceed this value.
        time_distances:
            time_distances between **all** the problem's locations, as an |array-like|_
            of positive numbers, including both customer sites and the depot.The first
            row and colum of the distance matrix are customer distances form the depot.
        time_window_open:
            The opening time of each customer, as an |array-like|_. The first element is
            the depot.
        time_window_close:
            The closing time of each customer, as an |array-like|_. The first element is
            the depot.
        service_time:
            The time it takes to serve each customer, as an |array-like|_. The first element
            is the depot and must be zero.

    Returns:
        A model encoding the CVRPTW problem.

    Notes:
        The model uses a :class:`~dwave.optimization.model.Model.disjoint_lists`
        class as the decision variable being optimized, with permutations of its
        sublist representing various itineraries for each vehicle.
    """

    if not isinstance(number_of_vehicles, int):
        raise ValueError("`number_of_vehicles` must be an integer.")

    if number_of_vehicles < 1:
        raise ValueError("`number_of_vehicles` must be at least 1."
                         f" Got {number_of_vehicles}.")

    if vehicle_capacity <= 0:
        raise ValueError("`vehicle_capacity` must be a positive number."
                         f" Got {vehicle_capacity}.")

    demand = _require("demand", demand, dtype=float, ndim=1, nonnegative=True)

    if demand.sum() > number_of_vehicles * vehicle_capacity:
        raise ValueError(f"Total demand, {demand.sum()}, exceeds the "
                         f"total capacity, {number_of_vehicles * vehicle_capacity}.")

    time_distances_array = _require("time_distances", time_distances, ndim=2, nonnegative=True, square=True)

    time_window_open = _require("time_window_open", time_window_open, dtype=float, ndim=1, nonnegative=True)
    time_window_close = _require("time_window_close", time_window_close, dtype=float, ndim=1, nonnegative=True)
    service_time = _require("service_time", service_time, dtype=float, ndim=1, nonnegative=True)

    if time_distances_array.shape[0] < 2:
        raise ValueError("Number of rows in `time_distances` must be at least 2."
                         f" Got {time_distances_array.shape[0]} rows.")

    if time_distances_array.shape[0] != len(demand):
        raise ValueError("Number of rows in `time_distances` and length of `demand` must be equal."
                         f" Got {time_distances_array.shape[0]} and"
                         f" {len(demand)}, respectively.")

    if time_distances_array.shape[0] != len(time_window_open):
        raise ValueError("Number of rows in `time_distances` and length of `time_window_open` must be equal."
                         f" Got {time_distances_array.shape[0]} and"
                         f" {len(time_window_open)}, respectively.")

    if time_distances_array.shape[0] != len(time_window_close):
        raise ValueError("Number of rows in `time_distances` and length of `time_window_close` must be equal."
                         f" Got {time_distances_array.shape[0]} and"
                         f" {len(time_window_close)}, respectively.")

    if time_distances_array.shape[0] != len(service_time):
        raise ValueError("Number of rows in `time_distances` and length of `service_time` must be equal."
                         f" Got  {time_distances_array.shape[0]} and"
                         f" {len(service_time)}, respectively.")

    if demand[0] != 0:
        raise ValueError("`demand[0]` must be zero.")

    customer_demand = demand[1:]

    if (customer_demand == 0).any():
        raise ValueError("Only the first element of `demand` can be zero."
                         f" Got zeros for indices {list(np.where(demand == 0)[0])}.")

    num_customers = len(customer_demand)

    # Construct the model
    model = Model()

    # Add the constants
    time_distance_ext = np.zeros((num_customers + 1, num_customers + 1))
    time_distance_ext[:num_customers, :num_customers] = time_distances_array[1:, 1:]

    # Time from depot to destination
    time_from_depo_ext = np.zeros(num_customers + 1)
    time_from_depo_ext[:num_customers] = time_distances_array[0, 1:]

    # Time from source to depot
    time_to_depo_ext = np.zeros(num_customers + 1)
    time_to_depo_ext[:num_customers] = time_distances_array[1:, 0]

    if np.equal(time_from_depo_ext, time_to_depo_ext).all():
        t_from_depo = t_to_depo = model.constant(time_from_depo_ext)
    else:
        t_from_depo = model.constant(time_from_depo_ext)
        t_to_depo = model.constant(time_to_depo_ext)

    t_cust = model.constant(time_distance_ext)

    time_open_ext = np.zeros(num_customers + 1)
    time_open_ext[:num_customers] = time_window_open[1:]
    t_open = model.constant(time_open_ext)

    time_close_ext = np.zeros(num_customers + 1)
    time_close_ext[:num_customers] = time_window_close[1:]
    time_close_ext[-1] = 1e6
    t_close = model.constant(time_close_ext)

    service_time_ext = np.zeros(num_customers + 1)
    service_time_ext[:num_customers] = service_time[1:]
    t_service = model.constant(service_time_ext)

    demand = model.constant(customer_demand)
    capacity = model.constant(vehicle_capacity)
    t_depo_close = model.constant(time_window_close[0])

    one = model.constant(1)

    # Add the decision variable
    routes_decision, routes = model.disjoint_lists(
        primary_set_size=num_customers,
        num_disjoint_lists=number_of_vehicles)

    # Capacity constraint
    capacity_constraints = [(demand[routes[vehicle_idx]].sum() <= capacity)
                            for vehicle_idx in range(number_of_vehicles)
                            ]
    for c in capacity_constraints:
        model.add_constraint(c)

    rh = np.arange(num_customers + 1).astype(int)
    range_helper = model.constant(rh)

    # this is number of clients each vehicle visits
    num_clients_in_route = {}
    for i in range(number_of_vehicles):
        num_clients_in_route[f'route{i}'] = routes[i].size()

    # Constrain the number of locations per route
    max_loc_per_route_constant = model.constant(3 * int(num_customers / number_of_vehicles))
    max_loc_per_route_constraints = [(num_clients_in_route[f'route{v}'] <= max_loc_per_route_constant)
                                     for v in range(number_of_vehicles)]
    for mlpr in max_loc_per_route_constraints:
        model.add_constraint(mlpr)

    # here we keep track of the time a vehicle arrives to each client
    # store them in the class to check the solution
    time_leaving = []
    serving_time = []
    times_back = []
    time_windows_constraints = []

    for vehicle_idx in range(number_of_vehicles):
        this_t_leaving = []
        this_t_windows_c = []
        this_t_serving_time = []

        for client_idx in range(max_loc_per_route_constant):

            # the condition here allows us to choose whether to choose a used index or to fall back to the last one
            # that points to a zero in the extended array
            condition = (num_clients_in_route[f'route{vehicle_idx}'] >= range_helper[client_idx] + one)

            if client_idx == 0:
                # index of the first customer visited, if length of the route is zero,
                # points to the last (0) element
                idx = where(condition, routes[vehicle_idx][:1].sum(), range_helper[-1])
                this_t_serving_time.append(maximum(t_from_depo[idx], t_open[idx]))
                this_t_leaving.append(this_t_serving_time[-1] + t_service[idx])

            else:
                # index of the previous customer visited
                previous_idx = where(condition, routes[vehicle_idx][client_idx - 1:client_idx].sum(), range_helper[-1])

                # index of the current customer visited
                idx = where(condition, routes[vehicle_idx][client_idx:client_idx + 1].sum(), range_helper[-1])

                this_t_serving_time.append(maximum(this_t_leaving[-1] + t_cust[previous_idx, idx], t_open[idx]))
                this_t_leaving.append(this_t_serving_time[-1] + t_service[idx])

            # adding constraint that there is enough time for servicing the client
            this_t_windows_c.append(this_t_leaving[-1] <= t_close[idx])

        condition = (num_clients_in_route[f'route{vehicle_idx}'] >= one)
        last_cust_idx = where(condition, routes[vehicle_idx][-1:].sum(), range_helper[-1])

        times_back.append(this_t_leaving[-1] + t_to_depo[last_cust_idx])
        time_leaving.append(this_t_leaving)
        serving_time.append(this_t_serving_time)

        for ct in this_t_windows_c:
            model.add_constraint(ct)
        time_windows_constraints.append(this_t_windows_c)

    times_back_constraints = [(tb <= t_depo_close) for tb in times_back]
    for c in times_back_constraints:
        model.add_constraint(c)

    # Add the objective
    # minimize travelled distance
    route_costs = []
    for r in range(number_of_vehicles):
        route_costs.append(t_from_depo[routes[r][:1]].sum())
        route_costs.append(t_to_depo[routes[r][-1:]].sum())
        route_costs.append(t_cust[routes[r][1:], routes[r][:-1]].sum())

    model.minimize(add(*route_costs))

    model.lock()

    return model


def flow_shop_scheduling(processing_times: numpy.typing.ArrayLike) -> Model:
    r"""Generate a model encoding a flow-shop scheduling problem.

    `Flow-shop scheduling <https://en.wikipedia.org/wiki/Flow-shop_scheduling>`_
    is a variant of the renowned :func:`.job_shop_scheduling` optimization problem.
    Given `n` jobs to schedule on `m` machines, with specified processing
    times for each job per machine, minimize the makespan (the total
    length of the schedule for processing all the jobs). For every job, the
    `i`-th operation is executed on the `i`-th machine. No machine can
    perform more than one operation simultaneously.

    `E. Taillard <http://mistic.heig-vd.ch/taillard/problemes.dir/problemes.html>`_
    provides benchmark instances compatible with this generator.

    .. Note::
        There are many ways to model flow-shop scheduling. The model returned
        by this function may or may not give the best performance for your
        problem.

    Args:
        processing_times:
            Processing times, as an :math:`m \times n` |array-like|_ of
            integers, where ``processing_times[m, n]`` is the time job
            `n` is on machine `m`.

    Returns:
        A model encoding the flow-shop scheduling problem.

    Examples:

        This example creates a model for a flow-shop-scheduling problem
        with three jobs on two machines. For example, the first job
        requires processing for 20 time units on the second machine in
        the flow of operations.

        >>> from dwave.optimization.generators import flow_shop_scheduling
        ...
        >>> processing_times = [[10, 5, 7], [20, 10, 15]]
        >>> model = flow_shop_scheduling(processing_times=processing_times)

    """
    processing_times = _require("processing_times", processing_times,
                                ndim=2, nonnegative=True)

    if not processing_times.size:
        raise ValueError("`processing_times` must not be empty")

    num_machines, num_jobs = processing_times.shape

    model = Model()

    # Add the constant processing-times matrix
    times = model.constant(processing_times)

    # The decision symbol is a num_jobs-long array of integer variables
    order = model.list(num_jobs)

    end_times = []
    for machine_m in range(num_machines):

        machine_m_times = []
        if machine_m == 0:

            for job_j in range(num_jobs):
                if job_j == 0:
                    machine_m_times.append(times[machine_m, :][order[job_j]])
                else:
                    end_job_j = times[machine_m, :][order[job_j]]
                    end_job_j += machine_m_times[-1]
                    machine_m_times.append(end_job_j)

        else:

            for job_j in range(num_jobs):
                if job_j == 0:
                    end_job_j = end_times[machine_m - 1][job_j]
                    end_job_j += times[machine_m, :][order[job_j]]
                    machine_m_times.append(end_job_j)
                else:
                    end_job_j = maximum(end_times[machine_m - 1][job_j], machine_m_times[-1])
                    end_job_j += times[machine_m, :][order[job_j]]
                    machine_m_times.append(end_job_j)

        end_times.append(machine_m_times)

    makespan = end_times[-1][-1]

    # The objective is to minimize the last end time
    model.minimize(makespan)

    model.lock()

    return model


def job_shop_scheduling(times: numpy.typing.ArrayLike, machines: numpy.typing.ArrayLike,
                        *,
                        upper_bound: typing.Optional[int] = None,
                        ) -> Model:
    """Generate a model encoding a job-shop scheduling problem.

    `Job-shop scheduling <https://en.wikipedia.org/wiki/Job-shop_scheduling>`_
    has many variant.  Here, what we have implemented is a variant of job-shop
    scheduling with the additional assumption that every job makes use of
    every machine.

    `E. Taillard <http://mistic.heig-vd.ch/taillard/problemes.dir/problemes.html>`_
    provides benchmark instances compatible with this generator.

    .. versionchanged:: 0.4.1
        Prior to version `0.4.1`, the model generated was based on one proposed by

        L. Blaise, "Modélisation et résolution de problèmes d’ordonnancement au
        sein du solveur d’optimisation mathématique LocalSolver", Université de
        Toulouse, https://hal-lirmm.ccsd.cnrs.fr/LAAS-ROC/tel-03923149v2.

        Now the model uses the more natural formulation where the only decision
        variables are the task start times, but with disjunctive non-overlapping
        constraints between each pair of job on the machines.

    .. Note::
        There are many ways to model job-shop scheduling. The model returned
        by this function may or may not give the best performance for your
        problem.

    Args:
        times:
            An ``n`` jobs by ``m`` machines array-like where ``times[n, m]``
            is the processing time of job ``n`` on machine ``m``.
        machines:
            An ``n`` jobs by ``m`` machines array-like where ``machines[n, :]``
            is the order of machines that job ``n`` will be processed on.
        upper_bound:
            An upper bound on the makespan.
            If not given, defaults to ``times.sum()``.
            Note that if the `upper_bound` is too small the model may be
            infeasible.

    Returns:
        A model encoding the job-shop scheduling problem.

    """
    times = _require("times", times, ndim=2, dtype=int, nonnegative=True)
    machines = _require("machines", machines, ndim=2, dtype=int, nonnegative=True)

    if times.shape != machines.shape:
        raise ValueError("`times` and `machines` must have the same shape")

    if not times.size:
        raise ValueError("`times` and `machines` must not be empty")

    num_jobs, num_machines = times.shape

    # Convert machines to 0-indexed if they are 1-indexed
    if machines.all():
        # in this case we're all non-zero, so let's assume (for the moment) that
        # we're range(1, num_jobs+1)
        # In that case we reindex to 0. This is OK because we know from our earlier
        # checking that we're non-negative, so combined with .all() we must be
        # strictly positive
        machines -= 1

        # the validity will be checked in the next for-loop

    # Each row of machines must be a permutation of range(num_machines) or of
    # range(1, num_machines+1)
    arange = np.arange(num_machines)
    for row in machines:
        if (np.sort(row) != arange).any():
            raise ValueError("each row of `machines` must be a permutation of "
                             "`range(num_machines)` or of `range(1, num_machines + 1)`")

    # Get the upper bound on the makespan. There are more clever ways to do this
    # but for now let's just take the loosest bound, which is the sum of all of
    # the times
    if upper_bound is None:
        upper_bound = times.sum()
    else:
        upper_bound = int(upper_bound)  # convert to an int
        if upper_bound < 1:
            raise ValueError("`upper_bound` must be a positive int")

    # Alright, model construction time
    model = Model()

    # Add the constants
    times = model.constant(times)
    machines = model.constant(machines)

    # The "main" decision symbol is a num_jobs x num_machines array of integer variables
    # giving the start time for each task
    start_times = model.integer(shape=(num_jobs, num_machines),
                                lower_bound=0, upper_bound=upper_bound)

    # The objective is simply to minimize the last end time
    end_times = start_times + times
    model.minimize(end_times.max())

    # Ensure that for each job, its tasks do not overlap
    for j in range(num_jobs):
        ends = end_times[j, machines[j, :-1]]
        starts = start_times[j, machines[j, 1:]]

        model.add_constraint((ends <= starts).all())

    # Ensure for each machine, its tasks do not overlap
    # Collect all the pairs of jobs in two indices arrays
    u_idx = []
    v_idx = []
    for i, j in itertools.combinations(range(num_jobs), 2):
        u_idx.append(i)
        v_idx.append(j)

    u = model.constant(u_idx)
    v = model.constant(v_idx)

    # Finally impose the non-overlapping constraints between jobs,
    # on all machines
    model.add_constraint(
        logical_or(
            end_times[u, :] <= start_times[v, :],
            end_times[v, :] <= start_times[u, :]
        ).all()
    )

    model.lock()
    return model


def knapsack(values: numpy.typing.ArrayLike,
             weights: numpy.typing.ArrayLike,
             capacity: float,
             ) -> Model:
    r"""Generate a model encoding a knapsack problem.

    The
    `knapsack problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_
    is, for a given set of items, each with a weight and a value, determine
    which items to include in the collection so that the total weight is less
    than or equal to a given limit and the total value is as large as possible.

    Args:
        values:
            A 1D |array-like|_ (row vector or Python list) of values per item.
            The length of ``values`` must be equal to the length of
            ``weights``. Values can be any non-negative number.

        weights:
            A 1D |array-like|_ (row vector or Python list) of weights per item.
            Weights can be any non-negative number.

        capacity:
            Maximum capacity of the knapsack. Must be a positive number.

    Returns:
        A model encoding the knapsack problem.

    The model generated uses a :class:`dwave.optimization.Model.set` class as
    the decision variable being optimized, with permutations of subsets of this
    set representing possible items included in the knapsack.

    """
    weights = _require("weights", weights, dtype=float, nonnegative=True, ndim=1)
    values = _require("values", values, dtype=float, nonnegative=True, ndim=1)

    num_items = weights.shape[0]

    if weights.shape != values.shape:
        raise ValueError("`weights` and `values` must be equal in length")

    if capacity <= 0:
        raise ValueError("`capacity` must be a positive number")

    # Construct the model
    model = Model()

    weights = model.constant(weights)
    # for minimization rather than maximization in the objective
    values = model.constant(-values)

    capacity = model.constant(capacity)

    items = model.set(num_items)

    # add the capacity constraint
    overcapacity = weights[items].sum() <= capacity
    model.add_constraint(overcapacity)

    # The objective is to maximize the value, which is.
    # equivalent to minimizing the negative values
    model.minimize(values[items].sum())

    model.lock()
    return model


def quadratic_assignment(distance_matrix: numpy.typing.ArrayLike,
                         flow_matrix: numpy.typing.ArrayLike,
                         ) -> Model:
    """Generate a model encoding a quadratic assignment problem.

    The
    `quadratic assignment <https://en.wikipedia.org/wiki/Quadratic_assignment_problem>`_
    is, for a given list of facilities, the distances between them, and the flow
    between each pair of facilities, minimize the sum of the products of distances
    and flows.

    Args:
        distance_matrix:
            An array-like where ``distance_matrix[n, m]`` is the distance
            between location ``n`` and ``m``. Represents the (known and constant)
            distances between every possible pair of facility locations: in real-world
            problems, such a matrix can be generated from an application with
            access to an online map.

        flow_matrix:
            A array-like where ``flow_matrix[n, m]`` is the flow
            between location ``n`` and ``m``. Represents the (known and constant)
            flow between every possible pair of facility location

    Returns:
        A model encoding the quadratic_assignment problem.

    """
    distance_matrix = _require("distance_matrix", distance_matrix,
                               dtype=float, ndim=2, nonnegative=True, square=True)
    flow_matrix = _require("flow_matrix", flow_matrix,
                           dtype=float, ndim=2, nonnegative=True, square=True)

    if distance_matrix.shape != flow_matrix.shape:
        raise ValueError("'distance_matrix' and 'flow_matrix' must have the same shape")

    # Construct the model
    qap_model = Model()

    # Represent the facility assignments efficiently using permutations of an ordered list
    num_facilities = distance_matrix.shape[0]
    ordered_facilities = qap_model.list(num_facilities)

    # Add the constants
    DISTANCE_MATRIX = qap_model.constant(distance_matrix)
    FLOW_MATRIX = qap_model.constant(flow_matrix)

    # Minimize the sum of distances multiplied by the flows.
    qap_model.minimize(
        (
                FLOW_MATRIX * DISTANCE_MATRIX[ordered_facilities, :][:, ordered_facilities]
        ).sum()
    )

    qap_model.lock()
    return qap_model


def traveling_salesperson(distance_matrix: numpy.typing.ArrayLike,
                          ) -> Model:
    r"""Generate a model encoding a traveling-salesperson problem.

    The
    `traveling salesperson <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_
    is, for a given a list of cities and distances between each pair of cities,
    to find the shortest possible route that visits each city exactly once and
    returns to the city of origin.

    Args:
        distance_matrix:
            An array-like where ``distance_matrix[n, m]`` is the distance
            between city ``n`` and ``m``. Represents the (known and constant)
            distances between every possible pair of cities: in real-world
            problems, such a matrix can be generated from an application with
            access to an online map.

    Returns:
        A model encoding the traveling-salesperson problem.

    Typically, solver performance strongly depends on the size of the solution
    space for your modelled problem: models with smaller spaces of feasible
    solutions tend to perform better than ones with larger spaces. A powerful
    way to reduce the feasible-solutions space is by using variables that act
    as implicit constraints. This is analogous to judicious typing of a variable
    to meet but not exceed its required assignments: a Boolean variable, ``x``,
    has a solution space of size 2 (:math:`\{True, False\}`) while an integer
    variable, ``i``, might have a solution space of over 4 billion values.

    The model generated uses a :class:`dwave.optimization.Model.list` class as
    the decision variable being optimized, with permutations of this ordered
    list representing possible itineraries through the required cities.

    The :class:`dwave.optimization.Model.list` class used to represent the
    decision variable as an ordered list is implicitly constrained in the
    possible values that can be assigned to it; for example, compare a model
    representing five cities as five variables of type ``int``,
    :math:`i_{Rome}, i_{Turin}, i_{Naples}, i_{Milan}, i_{Genoa}`, where
    :math:`i_{Rome} = 2` means Rome is the third city visited, versus a model
    using an ordered list, :math:`(city_0, city_1, city_2, city_3, city_4)`.
    The first model must explicitly constrain solutions to those that select
    a value between 0 to 4 for each decision variable with no repetitions; such
    constraints are implicit to the ordered-list variable.

    The objective is to minimize the distance traveled. Permutations of indices
    :math:`(0, 1, 2, ...)` that order ``distance_matrix`` represent itineraries
    that travel from list-element :math:`i` to list-element :math:`i+1` for
    :math:`i \in [0, N-1]`, where :math:`N` is the length of the list (and the
    number of cities). Additionally, the problem requires a return to the city
    of origin (first element of the list) from the last city visited (the last
    element of the list).

    """
    distance_matrix = _require("distance_matrix", distance_matrix,
                               dtype=float, ndim=2, nonnegative=True, square=True)

    # Construct the model
    tsp_model = Model()

    # Represent itineraries efficiently using permutations of an ordered list
    num_cities = distance_matrix.shape[0]
    ordered_cities = tsp_model.list(num_cities)

    # Add the constants
    DISTANCE_MATRIX = tsp_model.constant(distance_matrix)

    # The objective is to minimize the distance traveled.
    # The first elements of the pairs of cities traveled between are the
    # permuted indices excluding the last and the second elements are these
    # indices excluding the first.

    # Python represents this slicing of a list with slice indices ``[:-1]`` and ``[1:]``,
    # respectively, because Python slices are defined in the format of a
    # mathematical interval (see https://en.wikipedia.org/wiki/Interval_(mathematics))
    # [a, b) and list indices start at zero. For example, ``ordered_cities[:-1]``
    # excludes the last element of the sliced list.

    # Adding the return to the city of origin add the distance between the first
    # element of the list, or slice ``[0]``, and the last city visited, or slice
    # ``[-1]``

    itinerary = DISTANCE_MATRIX[ordered_cities[:-1], ordered_cities[1:]]
    return_to_origin = DISTANCE_MATRIX[ordered_cities[-1], ordered_cities[0]]

    # Sum the distances along the full route traveled.
    travel_distance = itinerary.sum() + return_to_origin.sum()

    # Minimize the total travel distance.
    tsp_model.minimize(travel_distance)

    tsp_model.lock()
    return tsp_model

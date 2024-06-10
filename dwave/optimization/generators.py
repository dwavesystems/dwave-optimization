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

import typing

import numpy as np
import numpy.typing

from dwave.optimization.mathematical import add, maximum 
from dwave.optimization.model import Model

__all__ = ["capacitated_vehicle_routing", "flow_shop_scheduling", "job_shop_scheduling", 
           "knapsack", "traveling_salesperson", ]


def _1d_real_array(**kwargs: numpy.typing.ArrayLike) -> typing.Iterator[np.ndarray]:
    """Coerce all given array-likes to 1D NumPy arrays and raise a consistent 
    error message if it cannot be cast.

    Keyword argument names must match the argument names of the calling function
    for the error message to make sense.
    """
    for argname, array in kwargs.items():
        try:
            array = np.asarray(array)
        except (ValueError, TypeError):
            raise ValueError(f"`{argname}` must be a 1D array-like of numbers")

        if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"`{argname}` must be a 1D array-like of numbers")

        yield array

def _1d_nonnegative_real_array(**kwargs: numpy.typing.ArrayLike) -> typing.Iterator[np.ndarray]:
    """Coerce all given array-likes to 1D NumPy arrays of non-negative numbers and
    raise a consistent error message if it cannot be cast.

    Keyword argument names must match the argument names of the calling function
    for the error message to make sense.
    """
    for argname, array in kwargs.items():
        try:
            array = np.asarray(array)
        except (ValueError, TypeError):
            raise ValueError(f"`{argname}` must be a 1D array-like of non-negative numbers")
        
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"`{argname}` must be a 1D array-like of non-negative numbers")

        if array.ndim != 1 or (array < 0).any():
            raise ValueError(f"`{argname}` must be a 1D array-like of non-negative numbers")

        yield array


def _2d_nonnegative_int_array(**kwargs: numpy.typing.ArrayLike) -> typing.Iterator[np.ndarray]:
    """Coerce all given array-likes to 2d NumPy arrays of non-negative integers and
    raise a consistent error message if it cannot be cast.

    Keyword argument names must match the argument names of the calling function
    for the error message to make sense.
    """
    for argname, array in kwargs.items():
        try:
            array = np.atleast_2d(np.asarray(array, dtype=int))
        except (ValueError, TypeError):
            raise ValueError(f"`{argname}` must be a 2d array-like of non-negative integers")
        
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"`{argname}` must be a 2d array-like of non-negative integers")

        if array.ndim != 2 or (array < 0).any():
            raise ValueError(f"`{argname}` must be a 2d array-like of non-negative integers")

        yield array


def _2d_nonnegative_symmetric_array(**kwargs: numpy.typing.ArrayLike) -> typing.Iterator[np.ndarray]:
    """Coerce all given array-likes to 2D NumPy arrays of non-negative floats
    or integers and raise a consistent error message if it cannot be cast.

    Keyword argument names must match the argument names of the calling function
    for the error message to make sense.
    """
    for argname, array in kwargs.items():
        try:
            array = np.atleast_2d(np.asarray(array))
        except (ValueError, TypeError):
            raise ValueError(f"`{argname}` must be a symmetric 2D array-like of non-negative numbers")
        
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"`{argname}` must be a symmetric 2D array-like of non-negative numbers")

        if array.shape[0] != array.shape[1] or (array < 0).any():
            raise ValueError(f"`{argname}` must be a symmetric 2D array-like of non-negative numbers")

        if not np.array_equal(array, array.T):
            raise ValueError(f"`{argname}` must be a symmetric 2D array-like of non-negative numbers")

        yield array


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
            Number of available vehicles, as an integer.   
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
    
    demand = next(_1d_nonnegative_real_array(demand=demand))

    if demand.sum() > number_of_vehicles*vehicle_capacity:
        raise ValueError(f"Total demand, {demand.sum()}, exceeds the "
                         f"total capacity, {number_of_vehicles*vehicle_capacity}.")

    if distances is not None:

        if depot_x_y is not None: 
            raise ValueError("`depot_x_y` and `distances` cannot be specified together.")
        
        distances_array = next(_2d_nonnegative_symmetric_array(distances=distances))

        if len(distances_array) < 2:
            raise ValueError("Length of `distances` must be at least 2."
                             f" Got length {len(distances_array)}.")

        if len(distances_array) != len(demand):
            raise ValueError("Lengths of `distances` and `demand` must be equal."
                             f" Got lengths {len(distances_array)} and"
                             f" {len(demand)}, respectively.")
        
        if demand[0] != 0:
            raise ValueError("`demand[0]` must be zero when `distances` is specified.")
        
        customer_demand = demand[1:]
        distance_matrix = distances_array[1:, 1:]
        depot_distance_vector = distances_array[0][1:]

    else:
        x, y = _1d_real_array(locations_x=locations_x,
                              locations_y=locations_y)
        
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

        depot_distance_vector = np.sqrt(
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
    demand = model.constant(customer_demand)
    capacity = model.constant(vehicle_capacity)

    # Add the decision variable
    routes_decision, routes = model.disjoint_lists(
        primary_set_size=num_customers,
        num_disjoint_lists=number_of_vehicles)

    # The objective is to minimize the distance traveled. 
    # This is calculated by adding the distance from the depot to the 1st customer
    # to the distance from the last customer back to the depot, using indices on
    # `depot_dist` selected by the choice of decision variable's routes, and to the 
    # distances between customers per route, using indices on `customer_dist` selected 
    # similarly.   
    # Python represents this slicing of a list with slice indices ``[:-1]`` and ``[1:]``, 
    # respectively, because Python slices are defined in the format of a
    # mathematical interval (see https://en.wikipedia.org/wiki/Interval_(mathematics)) 
    # [a, b) and list indices start at zero. For example, ``routes[r][:-1]`` 
    # excludes the last element of the sliced list. 
    route_costs = []
    for r in range(number_of_vehicles):
        route_costs.append(depot_dist[routes[r][:1]].sum())
        route_costs.append(depot_dist[routes[r][-1:]].sum())
        route_costs.append(customer_dist[routes[r][1:], routes[r][:-1]].sum())

        model.add_constraint(demand[routes[r]].sum() <= capacity)

    model.minimize(add(route_costs))

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
            Processing times, as an :math:`n \times m` |array-like|_ of 
            integers, where ``processing_times[n, m]`` is the time job 
            `n` is on machine `m`.
        
    Returns:
        A model encoding the flow-shop scheduling problem.
        
    Examples:
    
        This example creates a model for a flow-shop-scheduling problem
        with two jobs on three machines. For example, the second job 
        requires processing for 20 time units on the first machine in 
        the flow of operations.  
    
        >>> from dwave.optimization.generators import flow_shop_scheduling
        ...
        >>> processing_times = [[10, 5, 7], [20, 10, 15]]
        >>> model = flow_shop_scheduling(processing_times=processing_times)

    """
    processing_times = next(_2d_nonnegative_int_array(processing_times=processing_times))

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

    The model generated is based on the one proposed in
    L. Blaise, "Modélisation et résolution de problèmes d’ordonnancement au
    sein du solveur d’optimisation mathématique LocalSolver", Université de
    Toulouse, https://hal-lirmm.ccsd.cnrs.fr/LAAS-ROC/tel-03923149v2

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
    times, machines = _2d_nonnegative_int_array(times=times, machines=machines)

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

    # Each row of machines must be a permutation of range(num_jobs) or of range(1, num_jobs+1)
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

    # The "main" decision symbol is a num_machines x num_jobs array of integer variables
    # giving the start time for each task
    start_times = model.integer(shape=(num_machines, num_jobs),
                                lower_bound=0, upper_bound=upper_bound)

    # We also need a redundant decision symbol for each machine defining the
    # order of jobs on that machine
    orders = [model.list(num_jobs) for _ in range(num_machines)]

    # The objective is simply to minimize the last end time
    end_times = start_times + times
    model.minimize(end_times.max())

    # Ensure that for each job, its tasks do not overlap
    for j in range(num_jobs):
        ends = end_times[j, :][machines[j, :-1]]
        starts = start_times[j, :][machines[j, :]][1:]
        model.add_constraint((ends <= starts).all())

    # Ensure for each machine, its tasks do not overlap
    for m in range(num_machines):
        order = orders[m]

        ends = end_times[:, m]
        starts = start_times[:, m]

        # todo: with combined indexing we wouldn't need this loop
        for t in range(num_jobs - 1):
            end = ends[order[t]]  # end of the t'th job on machine m
            start = starts[order[t+1]]  # start of the (t+1)th job on machine m

            model.add_constraint(end <= start)

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
    weights, values = _1d_nonnegative_real_array(weights=weights,
                                                 values=values)

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
            A symmetric array-like where ``distance_matrix[n, m]`` is the distance 
            between city ``n`` and ``m``. Represents the (known and constant) 
            distances between every possible pair of cities: in real-world 
            problems, such a matrix can be generated from an application with 
            access to an online map. Note that ``distance_matrix`` is symmetric 
            because the distance between city ``n`` to city ``m`` is the same 
            regardless of the direction of travel.
        
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
    distance_matrix = next(_2d_nonnegative_symmetric_array(distance_matrix=distance_matrix))

    if distance_matrix.shape[0] <2:
        raise ValueError("`distance_matrix` must be at least 2x2")

    if np.diagonal(distance_matrix).any():
        raise ValueError("Diagonal values of `distance_matrix` must all be zero")

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
    travel_distance = itinerary.sum()
    travel_distance += return_to_origin.sum()
    
    # Minimize the total travel distance.
    tsp_model.minimize(travel_distance)
    
    tsp_model.lock()
    return tsp_model


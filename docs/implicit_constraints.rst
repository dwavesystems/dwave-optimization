.. _optimization_implicit_vars:

====================
Implicit Constraints
====================

When formulating optimization problems, the choice of decision variables
significantly impacts the model's clarity, size of the solution space, and,
ultimately, the solver's performance.

Ocean software's :class:`~dwave.optimization.model.Model` class provides several
specialized constructors for decision variables that encode "implicit
constraints." These :ref:`symbols <optimization_symbols>` inherently represent
common combinatorial structures such as permutations, subsets, or partitions,
guiding the solver to explore only valid configurations.

Using these implicitly constrained symbols offers several advantages:

*   **Simplified Model Formulation:** Complex constraints (e.g., ensuring
    all elements are unique and used in a sequence) are handled implicitly
    by the variable type itself, leading to more concise and readable
    models.

*   **Reduced Solution Space:** The solver's search space is drastically
    reduced because it only considers arrangements that satisfy the
    inherent nature of the symbol (e.g., permutations instead of all
    possible lists).

*   **Potential for Improved Performance:** A smaller, more structured search
    space can lead to faster solution times and better quality solutions.

This page details these specialized symbols, providing descriptions, creation
methods, practical examples, and common use cases. These symbols are directly
available as methods on the :class:`~dwave.optimization.model.Model` object.


.. _optimization_implicit_vars_comparison:

Summary Comparison
==================

To assist in selecting the implicitly constrained symbol most appropriate for a
given problem, the table below compares the key characteristics and typical
applications of each.

.. list-table:: Comparative Summary of Implicitly Constrained Symbols
    :widths: 15 20 20 22 22
    :header-rows: 1

    *   - **Feature**
        - ``list(N)``
        - ``set(N)``
        - ``disjoint_lists(...)``
        - ``disjoint_bit_sets(...)``
    *   - **Primary Purpose**
        - Ordered permutation of ``range(N)``
        - Unordered subset of ``range(N)``
        - Disjoint ordered partitions of ``range(primary_set_size)``
        - Disjoint unordered partitions of ``range(primary_set_size)``
    *   - **Order Within Group/List**
        - Yes
        - No
        - Yes (within each list)
        - No (within each set)
    *   - **Item Uniqueness**
        - All ``N`` items appear exactly once in the list
        - Unique subset from universe
        - Each item appears in at most one list; lists are permutations
        - Each item appears in at most one set; sets contain unique items
    *   - **Number of Collections**
        - 1 list
        - 1 set
        - ``num_disjoint_lists``
        - ``num_disjoint_sets``
    *   - **Creation Returns**
        - Single decision variable
        - Single decision variable
        - Main variable + collection of lists
        - Main variable + collection of sets
    *   - **Typical Problem Type**
        - TSP, QAP, sequencing
        - Knapsack, feature selection
        - CVRP, task assignment, multi-machine scheduling
        - Bin packing, clustering, set partitioning
    *   - **Input Parameters**
        - ``N``
        - ``N``
        - ``primary_set_size``, ``num_disjoint_lists``
        - ``primary_set_size``, ``num_disjoint_sets``


.. _optimization_implicit_vars_list:

Permutation
===========

The :meth:`~dwave.optimization.model.Model.list` constructor creates a decision
variable representing an ordered arrangement (a permutation) of :math:`N`
distinct items. These items are implicitly the integers
:math:`[0, 1, \ldots, N-1]`.

Overview
--------

.. table:: ``model.list(N)`` Overview
    :name: list_overview

    +----------------------+---------------------------------------------------+
    | **Feature**          | **Description**                                   |
    +======================+===================================================+
    | Conceptual Name      | :class:`~dwave.optimization.symbols.ListVariable` |
    |                      | (for understanding its role)                      |
    +----------------------+---------------------------------------------------+
    | Creation Method      | ``variable = model.list(N)``                      |
    +----------------------+---------------------------------------------------+
    | Purpose              | Represents an ordered sequence (permutation)      |
    |                      | of :math:`N` distinct items.                      |
    +----------------------+---------------------------------------------------+
    | Implicit Constraints | - All :math:`N` items (integers :math:`0` to      |
    |                      |   :math:`N-1`) are present exactly once.          |
    |                      |                                                   |
    |                      | - Order matters.                                  |
    |                      |                                                   |
    |                      | - Elements are unique.                            |
    +----------------------+---------------------------------------------------+
    | ``N`` Represents     | The number of items in the permutation.           |
    +----------------------+---------------------------------------------------+
    | Output in Solution   | A NumPy array of length :math:`N` containing      |
    |                      | a permutation of :math:`[0, 1, \ldots, N-1]`.     |
    |                      | May contain floats close to integers.             |
    +----------------------+---------------------------------------------------+

Description
-----------

:meth:`~dwave.optimization.model.Model.list` is ideal for problems where the
core decision involves finding the optimal order or sequence of a set of items.
Instead of creating :math:`N` integer variables and adding ``AllDifferent``
constraints along with range constraints, ``model.list(N)`` encapsulates
this. The solver explores the :math:`N!` possible permutations rather than the
:math:`N^N` combinations possible with :math:`N` unrestricted integer variables.
If your actual items are not integers :math:`0` to :math:`N-1` (e.g., city
names), you should map them to these integer indices before defining the model
and map them back when interpreting the solution. Note that the solver might
return indices as floats, requiring casting to ``int``.

Example: Traveling Salesperson Problem
--------------------------------------

The `traveling salesperson problem <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_
(TSP) requires finding the shortest possible route that visits each city exactly
once and returns to the origin city.

.. code:: python

    import dwave.optimization as do
    import numpy as np
    # Import the correct sampler from dwave.system
    from dwave.system import LeapHybridNLSampler

    # --- Problem Data ---
    city_names = ['A', 'B', 'C', 'D']
    num_cities = len(city_names) # This corresponds to N

    # Distances: A-A, A-B, A-C, A-D
    #            B-A, B-B, B-C, B-D
    #            ...
    distance_matrix_data = np.array([
        [0, 10, 15, 20],  # Distances from A
        [10, 0, 35, 25],  # Distances from B
        [15, 35, 0, 30],  # Distances from C
        [20, 25, 30, 0]   # Distances from D
    ])

    # --- Model Definition ---
    model = do.Model()

    # 'ordered_cities' will be a permutation of [0, 1, ..., num_cities-1]
    ordered_cities = model.list(num_cities) # N = num_cities

    # Add constants to the model
    DISTANCE_MATRIX = model.constant(distance_matrix_data)

    # --- Objective Function ---
    # Cost of legs between cities in the permuted order
    itinerary_cost = DISTANCE_MATRIX[ordered_cities[:-1], ordered_cities[1:]].sum()
    # Cost of returning from the last city to the first city
    return_to_origin_cost = DISTANCE_MATRIX[ordered_cities[-1], ordered_cities[0]].sum()
    total_travel_distance = itinerary_cost + return_to_origin_cost
    model.minimize(total_travel_distance)

    model.lock()
    print("--- model.list() Example: Traveling Salesperson Problem ---")
    print(f"Cities (mapped to indices 0-{num_cities-1}): {city_names}")
    print(f"Decision Variable: ordered_cities = model.list(N={num_cities})")

    # Example of solving using .state(0) (requires Leap account and environment configuration)
    try:
        # Instantiate the Leap Hybrid Nonlinear Sampler
        sampler = LeapHybridNLSampler()

        # Submit the model to the sampler
        results = sampler.sample(model, label='Example - TSP')

        # Wait for results if asynchronous (sampler might return a Future)
        if hasattr(results, 'result'): # Basic check if it might be a Future
                job_result_object = results.result() # Wait and get the actual results object.
                print(f"Future resolved.")
                # We assume this has implicitly populated the model state cache.
        else:
                job_result_object = results # Assume results are already available
                print(f"Synchronous result received.")

        # Now attempt to access the best state (index 0) via model symbols
        print("\n--- Solution (via model.state(0)) ---")
        # Using model.lock() based on user's provided analysis snippet
        with model.lock():
            try:
                objective_value = model.objective.state(0)
                print(f"Objective Value (State 0): {objective_value:.2f}")

                route_indices_float = ordered_cities.state(0) # Access state (might be float)
                # Cast indices to int before using them to index Python lists
                route_indices = [int(idx) for idx in route_indices_float]

                named_route = [city_names[idx] for idx in route_indices]
                named_route_loop = named_route + [named_route[0]]
                print(f"Optimal route indices (float): {route_indices_float}")
                print(f"Optimal route indices (int): {route_indices}")
                print(f"Optimal route: {' -> '.join(named_route_loop)}")

            except IndexError:
                    print("State 0 not found. Solver might have failed or returned no solutions.")
            except Exception as e_state:
                    print(f"Error accessing state 0: {e_state}")

    except Exception as e:
        print(f"\nSolver execution failed or requires configuration: {e}")

    # --- Solution (via model.state(0)) ---
    # Objective Value (State 0): 80.00
    # Optimal route indices (float): [3. 1. 0. 2.]
    # Optimal route indices (int): [3, 1, 0, 2]
    # Optimal route: D -> B -> A -> C -> D

Common Use Cases
----------------

*   **Traveling Salesperson Problem (TSP):** Finding the shortest tour.

*   **Quadratic Assignment Problem (QAP):** Assigning :math:`N` facilities
    to :math:`N` locations where the interaction cost depends on flow and
    distance, and the assignment is a permutation.

*   **Flow Shop Scheduling:** Determining the sequence of jobs on a series
    of machines to minimize makespan.

*   **Single Machine Scheduling:** Ordering tasks on a single resource.

*   Any problem requiring the determination of an optimal sequence or
    permutation.


.. _optimization_implicit_vars_set:

Subset
======

The :meth:`~dwave.optimization.model.Model.set` constructor creates a decision
variable representing an unordered collection (a subset) of unique items chosen
from a universe of :math:`N` items (integers :math:`0` to :math:`N-1`).

Overview
--------

.. table:: ``model.set(N)`` Overview
    :name: set_overview

    +----------------------+---------------------------------------------------+
    | **Feature**          | **Description**                                   |
    +======================+===================================================+
    | Conceptual Name      | :class:`~dwave.optimization.symbols.SetVariable`  |
    +----------------------+---------------------------------------------------+
    | Creation Method      | ``variable = model.set(N)``                       |
    +----------------------+---------------------------------------------------+
    | Purpose              | Represents an unordered subset of unique          |
    |                      | items chosen from a universe of :math:`N`         |
    |                      | items.                                            |
    +----------------------+---------------------------------------------------+
    | Implicit Constraints | - Elements selected are unique.                   |
    |                      |                                                   |
    |                      | - Order of elements within the set does not       |
    |                      |   matter.                                         |
    |                      |                                                   |
    |                      | - Items are chosen from the universe              |
    |                      |   :math:`[0, \ldots, N-1]`.                       |
    +----------------------+---------------------------------------------------+
    | ``N`` Represents     | The size of the universe from which items are     |
    |                      | chosen.                                           |
    +----------------------+---------------------------------------------------+
    | Output in Solution   | A NumPy array containing the unique integer       |
    |                      | indices of selected items, typically sorted.      |
    |                      | May contain floats close to integers.             |
    +----------------------+---------------------------------------------------+

Description
-----------

:meth:`~dwave.optimization.model.Model.set` is used when the decision involves
selecting a group of items, and the order of selection is irrelevant. The symbol
inherently handles the uniqueness of selected items. Constraints on the size
(cardinality) of the set or other properties based on the selected items are
typically added explicitly. As with
:meth:`~dwave.optimization.model.Model.list`, if the actual items are not
:math:`0` to :math:`N-1`, a mapping is necessary. Note that the solver might
return indices as floats, requiring casting to ``int``.

Example: Knapsack Problem
--------------------------

The `knapsack_problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_
requires that for a given a set of items, each with a weight and a value, you
determine which items to include in a collection so that the total weight is
less than or equal to a given limit (capacity) and the total value is maximized.

.. code:: python

    import dwave.optimization as do
    import numpy as np
    # Import the correct sampler from dwave.system
    from dwave.system import LeapHybridNLSampler 

    # --- Problem Data ---
    item_names = ['item0', 'item1', 'item2', 'item3', 'item4']
    num_items_universe = len(item_names) # This corresponds to N

    weights_data = np.array([10, 20, 30, 40, 50]) # Weight of each item
    values_data = np.array([60, 100, 120, 200, 210]) # Value of each item
    knapsack_capacity = 70 # Maximum weight the knapsack can hold

    # --- Model Definition ---
    model = do.Model()

    # 'selected_items' will be a subset of [0, 1, ..., num_items_universe-1]
    selected_items = model.set(num_items_universe) # N = num_items_universe

    # Add constants
    WEIGHTS = model.constant(weights_data)
    VALUES = model.constant(values_data)
    CAPACITY = model.constant(knapsack_capacity)

    # --- Constraints ---
    # The sum of weights of selected items must not exceed capacity.
    total_weight_of_selected = WEIGHTS[selected_items].sum()
    model.add_constraint(total_weight_of_selected <= CAPACITY, label="capacity_constraint")

    # --- Objective Function ---
    # Maximize the total value of selected items.
    total_value_of_selected = VALUES[selected_items].sum()
    model.maximize(total_value_of_selected)

    model.lock()
    print("\n--- model.set() Example: Knapsack Problem ---")
    print(f"Universe of items (indices 0-{num_items_universe-1}): {item_names}")
    print(f"Decision Variable: selected_items = model.set(N={num_items_universe})")

    # Example of solving using .state(0) (requires Leap account and environment configuration)
    try:
        # Instantiate the Leap Hybrid Nonlinear Sampler
        sampler = LeapHybridNLSampler()

        # Submit the model to the sampler
        results = sampler.sample(model, label='Example - Knapsack')

        # Wait for results if asynchronous
        if hasattr(results, 'result'):
            job_result_object = results.result()
            print(f"Future resolved.")
        else:
            job_result_object = results
            print(f"Synchronous result received.")

        # Access the best state (index 0) via model symbols
        print("\n--- Solution (via model.state(0)) ---")
        with model.lock():
            try:
                # Maximization objective value might need interpretation from energy
                objective_value = model.objective.state(0)
                print(f"Objective Value (State 0 - check interpretation): {objective_value}")

                chosen_item_indices_float = selected_items.state(0) # Access state (might be float)
                # Cast indices to int for processing
                chosen_item_indices = [int(idx) for idx in chosen_item_indices_float]

                chosen_item_names = [item_names[idx] for idx in chosen_item_indices]
                print(f"Selected item indices (float): {chosen_item_indices_float}")
                print(f"Selected item indices (int): {chosen_item_indices}")
                print(f"Selected items: {chosen_item_names}")

                # Re-calculate value and weight from solution indices for clarity
                # Use the integer indices for NumPy array indexing
                actual_value = values_data[chosen_item_indices].sum()
                actual_weight = weights_data[chosen_item_indices].sum()
                print(f"Recalculated Value: {actual_value}")
                print(f"Recalculated Weight: {actual_weight} (Capacity: {knapsack_capacity})")

            except IndexError:
                    print("State 0 not found. Solver might have failed or returned no solutions.")
            except Exception as e_state:
                    print(f"Error accessing state 0: {e_state}")

    except Exception as e:
        print(f"\nSolver execution failed or requires configuration: {e}")

    # --- Solution (via model.state(0)) ---
    # Objective Value (State 0 - check interpretation): 360.0
    # Selected item indices (float): [0. 1. 3.]
    # Selected item indices (int): [0, 1, 3]
    # Selected items: ['item0', 'item1', 'item3']
    # Recalculated Value: 360
    # Recalculated Weight: 70 (Capacity: 70)

Common Use Cases
----------------

*   **Knapsack Problem:** Selecting items to maximize value/utility within
    a budget/capacity.

*   **Set-Covering, Packing, and Partitioning Problems:** Selecting subsets
    to satisfy coverage or disjointness requirements.

*   **Feature Selection:** Choosing a subset of features in machine
    learning.

*   **Committee Selection:** Forming a team or committee with specific
    properties from a larger pool.

*   Resource allocation problems where a selection of resources is needed.


.. _optimization_implicit_vars_disjoint_lists:

Disjoint Ordered Lists
======================

The :meth:`~dwave.optimization.model.Model.disjoint_lists` constructor creates a
complex decision variable. It partitions items from a primary set (integers
:math:`0` to ``primary_set_size-1``) into a specified number of lists,
``num_disjoint_lists``. Each of these lists is an ordered sequence (permutation)
of a subset of the primary set, and no item from the primary set can appear in
more than one list.

Overview
---------

.. table:: ``model.disjoint_lists(primary_set_size, num_disjoint_lists)`` Overview
    :name: disjoint_lists_overview

    +------------------------+---------------------------------------------------------------------+
    | **Feature**            | **Description**                                                     |
    +========================+=====================================================================+
    | Conceptual Name        | :class:`~dwave.optimization.symbols.DisjointLists`                  |
    +------------------------+---------------------------------------------------------------------+
    | Creation Method        | ``decision_var, list_collection =                                   |
    |                        | model.disjoint_lists(primary_set_size, num_disjoint_lists)``        |
    +------------------------+---------------------------------------------------------------------+
    | Purpose                | Partitions items from a primary set into several mutually exclusive |
    |                        | ordered lists.                                                      |
    +------------------------+---------------------------------------------------------------------+
    | Implicit Constraints   | - Each item from the primary set (indices :math:`0` to              |
    |                        |   ``primary_set_size-1``) appears in at most one list.              |
    |                        |                                                                     |
    |                        | - Order matters within each list.                                   |
    |                        |                                                                     |
    |                        | - Lists are disjoint regarding item membership.                     |
    +------------------------+---------------------------------------------------------------------+
    | ``primary_set_size``   | The number of unique items in the overall pool (universe            |
    |                        | ``range(primary_set_size)``) to be distributed and ordered.         |
    +------------------------+---------------------------------------------------------------------+
    | ``num_disjoint_lists`` | The number of separate, ordered lists to create.                    |
    +------------------------+---------------------------------------------------------------------+
    | Output in Solution     | ``list_collection`` provides access to the individual lists. Each   |
    |                        | list (e.g., ``list_collection[i]``) is a NumPy array of ordered item|
    |                        | indices. May contain floats close to integers. Accessing state      |
    |                        | requires care (see example).                                        |
    +------------------------+---------------------------------------------------------------------+

Description
-----------

This symbol is exceptionally powerful for problems like vehicle routing, where a
set of customers needs to be divided among several vehicles, and each vehicle
follows a specific ordered route. The ``list_collection`` object returned allows
you to access and constrain each list individually (e.g., ``list_collection[0]``
for the first vehicle's route). Note that the solver might return indices as
floats, requiring casting to ``int``.

Example: Capacitated Vehicle Routing Problem
--------------------------------------------

The `capacitated vehicle routing problem <https://en.wikipedia.org/wiki/Vehicle_routing_problem>`_
(simplified CVRP) is to assign customers to vehicles, where each vehicle has a
capacity, and to minimize total distance. Each vehicle follows an ordered route.

.. code:: python

    import dwave.optimization as do
    import numpy as np
    # Import the correct sampler from dwave.system
    from dwave.system import LeapHybridNLSampler

    # --- Problem Data ---
    num_customers = 5 # This is primary_set_size
    num_vehicles = 2  # This is num_disjoint_lists
    customer_demands_data = np.array([10, 15, 8, 12, 20])
    vehicle_capacity = 35

    # --- Model Definition ---
    model = do.Model()

    # routes_decision is the core variable.
    # routes provides accessors like routes[k] which are symbolic lists.
    routes_decision, routes = model.disjoint_lists(
        primary_set_size=num_customers,
        num_disjoint_lists=num_vehicles
    )

    DEMANDS = model.constant(customer_demands_data)
    CAPACITY = model.constant(vehicle_capacity)

    # --- Constraints ---
    all_route_costs = []
    for k in range(num_vehicles):
        vehicle_route_k = routes[k] # Symbolic representation of the k-th route
        demand_on_route_k = DEMANDS[vehicle_route_k].sum()
        model.add_constraint(demand_on_route_k <= CAPACITY, label=f"capacity_vehicle_{k}")

        num_cust_on_route_k = vehicle_route_k.size()
        # Placeholder cost: a real model uses distance matrices
        cost_for_route_k = num_cust_on_route_k
        all_route_costs.append(cost_for_route_k)

    # --- Objective Function ---
    total_cost = do.add(*all_route_costs)
    model.minimize(total_cost)

    model.lock()
    print("\n--- model.disjoint_lists() Example: Simplified Vehicle Routing ---")
    print(f"Customers (indices 0-{num_customers-1}), Vehicles: {num_vehicles}")
    print(f"Decision: routes_decision, routes = model.disjoint_lists(primary_set_size={num_customers}, num_disjoint_lists={num_vehicles})")

    # Example of solving using .state(0) (requires Leap account and environment configuration)
    try:
        # Instantiate the Leap Hybrid Nonlinear Sampler
        sampler = LeapHybridNLSampler()

        # Submit the model to the sampler
        results = sampler.sample(model, label='Example - CVRP (Simplified)')

        # Wait for results if asynchronous
        if hasattr(results, 'result'):
            job_result_object = results.result()
            print(f"Future resolved.")
        else:
            job_result_object = results
            print(f"Synchronous result received.")

        # Access the best state (index 0) via model symbols
        print("\n--- Solution (via model.state(0)) ---")
        with model.lock():
            try:
                objective_value = model.objective.state(0)
                print(f"Objective Value (State 0 - placeholder cost): {objective_value:.2f}")

                # Accessing state for collection: Try getting state for each sub-list
                print("CVRP Routes (State 0):")
                resolved_routes = []
                for v_idx in range(num_vehicles):
                    # Attempt to get state of the symbolic list routes[v_idx]
                    route_indices_float = routes[v_idx].state(0)
                    # Cast indices to int
                    route_indices = [int(idx) for idx in route_indices_float]
                    resolved_routes.append(route_indices)
                    print(f"  Vehicle {v_idx} route (indices): {route_indices}")
                    if len(route_indices) > 0:
                        # Use integer indices for NumPy indexing
                        route_demands = customer_demands_data[route_indices].sum()
                        print(f"    Demand: {route_demands} (Capacity: {vehicle_capacity})")

            except IndexError:
                    print("State 0 not found. Solver might have failed or returned no solutions.")
            except Exception as e_state:
                    print(f"Error accessing state 0: {e_state}")

    except Exception as e:
        print(f"\nSolver execution failed or requires configuration: {e}")

    # --- Solution (via model.state(0)) ---
    # Objective Value (State 0 - placeholder cost): 5.00
    # CVRP Routes (State 0):
    #   Vehicle 0 route (indices): [0, 2, 3]
    #     Demand: 30 (Capacity: 35)
    #   Vehicle 1 route (indices): [1, 4]
    #     Demand: 35 (Capacity: 35)

Common Use Cases
----------------

*   **Vehicle Routing Problems (CVRP, CVRPTW):** Assigning customers to
    vehicles and determining the optimal sequence of visits for each vehicle.

*   **Multi-Agent Task Assignment and Scheduling:** Allocating tasks to
    different agents/robots where each agent performs a sequence of assigned
    tasks.

*   **Parallel Machine Scheduling:** Assigning jobs to different machines and
    sequencing them on each machine.


.. _optimization_implicit_vars_disjoint_bit_sets:

Disjoint Unordered Sets
=======================

The :meth:`~dwave.optimization.model.Model.disjoint_bit_sets` constructor is
used to partition a universe of ``primary_set_size`` items (integers :math:`0`
to ``primary_set_size-1``) into ``num_disjoint_sets`` mutually exclusive,
unordered sets.

Overview
--------

.. table:: ``model.disjoint_bit_sets(primary_set_size, num_disjoint_sets)`` Overview
    :name: disjoint_bit_sets_overview

    +--------------------------+-------------------------------------------------------------------+
    | **Feature**              | **Description**                                                   |
    +==========================+===================================================================+
    | Conceptual Name          | :class:`~dwave.optimization.symbols.DisjointBitSet`               |
    +--------------------------+-------------------------------------------------------------------+
    | Creation Method          | ``decision_var, set_collection =                                  |
    |                          | model.disjoint_bit_sets(primary_set_size, num_disjoint_sets)``    |
    +--------------------------+-------------------------------------------------------------------+
    | Purpose                  | Partitions items from a universe into several mutually exclusive, |
    |                          | unordered sets.                                                   |
    +--------------------------+-------------------------------------------------------------------+
    | Implicit Constraints     | - Each item from the universe (indices :math:`0` to               |
    |                          |   ``primary_set_size-1``) appears in at most one set.             |
    |                          |                                                                   |
    |                          | - Order does not matter within each set.                          |
    |                          |                                                                   |
    |                          | - Sets are disjoint.                                              |
    +--------------------------+-------------------------------------------------------------------+
    | ``primary_set_size`` is  | The number of unique items (universe ``range(primary_set_size)``) |
    |                          | in the overall pool to be distributed.                            |
    +--------------------------+-------------------------------------------------------------------+
    | ``num_disjoint_sets`` is | The number of separate, unordered sets (e.g., bins, clusters) to  |
    |                          | create.                                                           |
    +--------------------------+-------------------------------------------------------------------+
    | Output in Solution       | ``set_collection`` provides access to individual sets. Each set   |
    |                          | (e.g., ``set_collection[i]``) is a NumPy array of unique,         |
    |                          | unordered item indices. May contain floats close to integers.     |
    |                          | Accessing state requires care (see example).                      |
    +--------------------------+-------------------------------------------------------------------+


Description
-----------

This symbol is suited for problems where items need to be grouped into distinct
categories or containers, and the order of items within a category does not
matter. The ``set_collection`` object allows individual manipulation and
constraint of each set (e.g., ``set_collection[0]`` for the first bin's
contents). The items to be partitioned are integers from
``range(primary_set_size)``. Note that the solver might return indices
as floats, requiring casting to ``int``.

Example: Bin Packing Problem
----------------------------

The `bin packing problem <https://en.wikipedia.org/wiki/Bin_packing_problem>`_
is, for a given a set of items with specified weights, to pack them into the
minimum number of bins, each with a fixed capacity.

.. code:: python

    import dwave.optimization as do
    import numpy as np
    # Import the correct sampler from dwave.system
    from dwave.system import LeapHybridNLSampler
    # Import the symbolic 'add' function
    from dwave.optimization.mathematical import add # Keep 'where' import if needed elsewhere

    # --- Problem Data ---
    item_weights_data = np.array([4, 8, 1, 4, 2, 1])
    num_items_to_pack = len(item_weights_data) # This is 'primary_set_size'
    bin_capacity = 10
    max_possible_bins = num_items_to_pack    # This is 'num_disjoint_sets'

    # --- Model Definition ---
    model = do.Model()

    # main_decision is the variable; bins_collection allows access to each set
    main_decision, bins_collection = model.disjoint_bit_sets(
        primary_set_size=num_items_to_pack,
        num_disjoint_sets=max_possible_bins
    )

    WEIGHTS = model.constant(item_weights_data)
    CAPACITY = model.constant(bin_capacity)
    ONE = model.constant(1)
    ZERO = model.constant(0)

    # --- Constraints ---
    # Each bin's total weight must not exceed capacity
    for i in range(max_possible_bins):
        bin_i_contents = bins_collection[i] # Symbolic representation of items in bin i
        weight_in_bin_i = WEIGHTS[bin_i_contents].sum()
        model.add_constraint(weight_in_bin_i <= CAPACITY) # Removed label

    # --- Objective Function (Mirroring generator logic) ---
    # Minimize the number of bins used
    num_bins_used_symbol = model.constant(0.0) # Initialize as float constant

    for i in range(max_possible_bins):
        # Assume .sum() gives symbolic count (size) of items in bin i
        symbolic_size = bins_collection[i].sum()
        # Condition: bin contains at least one item (Symbolic Boolean)
        is_bin_i_used = (symbolic_size >= ONE)
        # Add the symbolic boolean directly to the objective accumulator
        num_bins_used_symbol = num_bins_used_symbol + is_bin_i_used

    # Minimize the resulting symbolic sum
    model.minimize(num_bins_used_symbol)

    model.lock()
    # Update print statement to reflect correct parameter names used
    print("\n--- model.disjoint_bit_sets() Example: Bin Packing Problem ---")
    print(f"Items (indices 0-{num_items_to_pack-1}) with weights: {item_weights_data}")
    print(f"Decision: main_decision, bins_collection = model.disjoint_bit_sets(primary_set_size={num_items_to_pack}, num_disjoint_sets={max_possible_bins})")

    # Example of solving using .state(0) (requires Leap account and environment configuration)
    try:
        # Instantiate the Leap Hybrid Nonlinear Sampler
        sampler = LeapHybridNLSampler()

        # Submit the model to the sampler
        results = sampler.sample(model, label='Example - Bin Packing')

        # Wait for results if asynchronous
        if hasattr(results, 'result'):
            job_result_object = results.result()
            print(f"Future resolved.")
        else:
            job_result_object = results
            print(f"Synchronous result received.")

        # Access the best state (index 0) via model symbols
        print("\n--- Solution (via model.state(0)) ---")
        with model.lock():
            try:
                objective_value = model.objective.state(0)
                print(f"Minimum bins used (State 0): {objective_value:.0f}")

                # Accessing state for collection:
                for b_idx in range(max_possible_bins):
                    # Attempt to get state of the symbolic set bins_collection[b_idx]
                    # This returns the BIT VECTOR representation (array of 0s/1s, possibly float)
                    bin_contents_bit_vector = bins_collection[b_idx].state(0)

                    # Convert bit vector to list of integer indices
                    # Find indices where the value is close to 1 (handle potential floats)
                    indices_where_one = np.where(np.array(bin_contents_bit_vector) > 0.5)[0]
                    # Cast these indices to int
                    bin_contents_indices = [int(idx) for idx in indices_where_one]

                    if len(bin_contents_indices) > 0: # Only print used bins
                        # Use the derived integer indices for NumPy indexing
                        bin_item_weights = item_weights_data[bin_contents_indices]
                        actual_bin_weight = bin_item_weights.sum()
                        # Check if capacity constraint holds for this state
                        violation_flag = "*" if actual_bin_weight > bin_capacity else ""
                        print(f"  Bin {b_idx} (item indices): {bin_contents_indices}")
                        print(f"    Weights: {bin_item_weights}, Sum: {actual_bin_weight} (Capacity: {bin_capacity}) {violation_flag}")

            except IndexError:
                    print("State 0 not found. Solver might have failed or returned no solutions.")
            except Exception as e_state:
                    print(f"Error accessing state 0: {e_state}")

    except Exception as e:
        print(f"\nSolver execution failed or requires configuration: {e}")

    # --- Solution (via model.state(0)) ---
    # Minimum bins used (State 0): 3
    #   Bin 0 (item indices): [0, 2, 4, 5]
    #     Weights: [4 1 2 1], Sum: 8 (Capacity: 10)
    #   Bin 1 (item indices): [1]
    #     Weights: [8], Sum: 8 (Capacity: 10)
    #   Bin 4 (item indices): [3]
    #     Weights: [4], Sum: 4 (Capacity: 10)

Common Use Cases
----------------

*   **Bin Packing:** Assigning items to a minimum number of bins.

*   **Set-Partitioning and Clustering:** Dividing items into disjoint,
    unordered groups.

*   **Resource Allocation:** Grouping resources into pools where order
    within a pool doesn’t matter.

*   **Graph Coloring (Vertex Coloring Variant):** Assigning vertices
    (items) to color classes (sets) such that no two adjacent vertices
    share the same color.


.. _optimization_implicit_vars_guidelines:

Selection Guidelines
====================

*   **Identify Core Combinatorial Structure:** Analyze your problem. Does
    it fundamentally involve:

    -   Finding an optimal order/sequence? :math:`\rightarrow`
        ``model.list(N)``

    -   Choosing a group of items without order? :math:`\rightarrow`
        ``model.set(N)``

    -   Partitioning items into distinct ordered sequences?
        :math:`\rightarrow`
        ``model.disjoint_lists(primary_set_size, num_disjoint_lists)``

    -   Grouping items into distinct unordered collections?
        :math:`\rightarrow`
        ``model.disjoint_bit_sets(primary_set_size, num_disjoint_sets)``

*   **Prefer Implicit Constraints:** When a specialized symbol naturally
    fits the problem’s structure, prefer it over defining basic variables
    (like ``model.integer()`` or ``model.binary()``) and then adding many
    explicit constraints (e.g., ``AllDifferent``, pairwise inequalities
    for ordering, etc.). This often leads to more robust and performant
    models.

*   **Mapping to Indices:** Remember that these symbols operate on integer
    indices (e.g., :math:`0` to :math:`N-1`, :math:`0` to ``size-1``,
    etc.). If your problem involves named items or other data types,
    create a mapping to these indices before model construction and map
    the solution indices back to your original item identifiers for
    interpretation.

*   **Start Simple:** If unsure, start with the symbol that seems most
    appropriate. You can always refine or change the model structure if
    needed. The :ref:`optimization_generators` source code provides
    excellent examples of these symbols in action for classic problems.

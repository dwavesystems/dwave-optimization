.. _optimization_philosophy:

.. |TM| replace:: :sup:`TM`

==========
Philosophy
==========

`dwave-optimization` and
the `Leap <https://cloud.dwavesys.com/leap/>`_\ |TM| service's quantum-classical 
hybrid nonlinear solver
incorporate features and design principals from each of the following areas:

.. _optimization_philosophy_quantum_optimization:

Quantum Optimization
====================

:term:`Quantum computing` has the potential to help solve some of the most complex
technical, scientific, national defense, and commercial problems that
organizations face.

:term:`Quantum annealing` processors naturally return low-energy solutions.
A fundamental rule of physics is that everything tends to seek a minimum energy state.
Objects slide down hills; hot things cool down over time.
This behavior is also true in the world of quantum physics.
Quantum annealing simply uses quantum physics to find low-energy states of a
problem and therefore the optimal or near-optimal combination of elements.

To learn more about solving optimization problems with quantum computers, see
:ref:`qpu_index_get_started`.

.. _optimization_philosophy_linear_programming:

(Mixed-Integer) Linear Programming
==================================

Much of the mathematical optimization done today is done with
`linear programming <https://en.wikipedia.org/wiki/Linear_programming>`_
solvers and techniques.

Users familiar with linear programming techniques will find many concepts in
`dwave-optimization` familiar: objective functions; continuous, linear, and binary variables;
constraints; and feasible regions are all concepts found in linear programming.

However, unlike linear solvers, `dwave-optimization` allows
`nonlinear <https://en.wikipedia.org/wiki/Nonlinear_programming>`_ relationships
among variables, constraints, and the objective function.

.. _optimization_philosophy_combinatorial_variables:

Lists, Sets, and other combinatorial variables
==============================================

While many optimization problems can be expressed using only scalar variable
types (continuous, integer, and binary), it is often convenient to use other
combinatorial structures.

In the following example, we will contrast a classic mixed integer linear programming
formulation using scalar variables with one that uses a list variable.

Example - Traveling Salesperson
-------------------------------

The goal of renowned
`traveling salesperson problem <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_
is, for a given a list of cities and distances between each pair of cities, to
find the shortest possible route that visits each city exactly once and returns
to the city of origin.

If we wish to restrict ourselves to scalar variables,
a common mixed-integer linear programming formulation for solving a TSP with
:math:`N` cities is to use :math:`N^2` binary variables and :math:`2N`
constraints.

.. dropdown:: MILP formulation details

    Let

    .. math::
        
        x_{it} =
            \begin{cases} 
                1 \text{ if the salesperson is in city } i \text{ at time } t \\
                0 \text{ otherwise} \\
            \end{cases}

    If :math:`d_{i,j}` gives the distance from city :math:`i` to city :math:`j` then
    the objective is to minimize the total tour length

    .. math::

        \sum_{st} \sum_{ij} x_{is} x_{jt} d_{ij}

    subject to constraints that the salesperson cannot be in two cities
    at the same time, nor any city twice

    .. math::

        \sum_i x_{it} = 1 \text{  } \forall t \\
        \sum_t x_{it} = 1 \text{  } \forall i

However, the solution to this problem is defined by an ordered list of cities.
For example, a three city traveling salesperson problem has
`(0, 1, 2)`, `(0, 2, 1)`, `(1, 0, 2)`, `(1, 2, 0)`, `(2, 0, 1)`, and `(2, 1, 0)`
as possible solutions.
Therefore, rather than thinking about scalars, we can define a *list* variable as
any permutation of the integers from :math:`[0..N)`.
Another advantage of this approach is that the problem is unconstrained!

.. dropdown:: Formulation with a list variable details

    Let `x` be a list variable of length `N` with `x_t` giving
    the city the salesperson visits at time `t`. Let's define `x_N = x_0` for
    convenience.

    If :math:`d_{i,j}` gives the distance from city :math:`i` to city :math:`j` then
    the objective is to minimize the total tour length

    .. math::

        \sum_t d_{x_t, x_{t+1}}

    .. seealso:: :func:`~dwave.optimization.generators.traveling_salesperson`
        A generator encoding this formulation.

We can compare the MILP and the list formulations by the number of constraints
and the size of their search space.

.. csv-table::
   :header-rows: 2

   , MIQP, , Nonlinear
   # of Facilities, Variable Domain Size, # of Constraints, Variable Domain Size, # of Constraints
   N, :math:`2^{N^2}`, :math:`2N`, :math:`N!`, 0
   5, 33554432, 10, 120, 0
   10, 1267650600228229401496703205376, 20, 3628800, 0

By using the list formulation, the search space is much smaller and
the solver is more likely to be able to find an optimal solution.

.. _optimization_philosophy_tensor_programming:

Tensor Programming
==================

`NumPy <https://numpy.org/>`_ is the most popular scientific computing library
in Python. The NumPy library contains data structures for multidimensional arrays.
To learn more, NumPy provides an excellent
`introduction to arrays <https://numpy.org/doc/stable/user/absolute_beginners.html>`_.

Working with arrays can be beneficial for readability and for performance.
Consider calculating the dot product of two lists of numbers ``a`` and ``b``.
This can be accomplished in Python with

.. code-block:: python

    a = [...]
    b = [...]
    value = sum(u * b for u, v in zip(a, b))

But it is more readable and more performant to express it using array operations

.. code-block:: python

    a = np.asarray([...])
    b = np.asarray([...])

    value = (a * b).sum()  # or even np.dot(a, b)

*dwave-optimization* can be thought of as a framework for symbolically encoding
operations over multidimensional arrays. It therefore inherits much of its API
as well as performance intuition from NumPy. With *dwave-optimization* the
previous example can be expressed as

.. code-block:: python

    model = dwave.optimization.Model()

    a = model.constant([...])
    b = model.constant([...])

    value = (a * b).sum()

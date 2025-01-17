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

However, unlike other linear solvers on the market, `dwave-optimization` allows
`nonlinear <https://en.wikipedia.org/wiki/Nonlinear_programming>`_ relationships
among variables, constraints, and the objective function.

.. _optimization_philosophy_constraint_programming:

Lists, Sets, and other combinatorial variables
==============================================

Lorem ipsum

.. _optimization_philosophy_tensor_programming:

Tensor Programming
==================

Lorem ipsum


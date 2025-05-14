.. _index_optimization:

==================
dwave-optimization
==================

.. toctree::
    :caption: Reference documentation for dwave-optimization:
    :maxdepth: 1

    api_ref

About dwave-optimization
========================

.. include:: README.rst
  :start-after: start_optimization_about
  :end-before: end_optimization_about

For explanations of the terminology, see the
:ref:`Concepts <index_concepts>` section.

Design Principals
=================

.. toctree::
  :maxdepth: 1
  :hidden:

  philosophy

`dwave-optimization` and the hybrid nonlinear solver incorporate features and
:ref:`design principals <optimization_philosophy>` from each of the following
areas:

.. developer note: this will look funny when built locally but looks fine as
   part of the sdk

.. grid:: 2 2 3 2
    :gutter: 2

    .. grid-item-card:: Quantum Optimization
        :img-top: /_images/optimization_quantum_icon.svg
        :link: optimization_philosophy_quantum_optimization
        :link-type: ref

        Take advantage of quantum-mechanical effects not available to classical
        compute.

    .. grid-item-card:: (Mixed-Integer) Linear Programming
        :img-top: /_images/optimization_lp_icon.svg
        :link: optimization_philosophy_linear_programming
        :link-type: ref

        Learn the basics of solving optimization problems with linear program
        solvers.

    .. grid-item-card:: Lists, Sets, and Other Combinatorial Variables
        :img-top: /_images/optimization_combinatorial_icon.svg
        :link: optimization_philosophy_combinatorial_variables
        :link-type: ref

        Explore how lists, sets, and other combinatorial structures make
        optimization simpler and more performant.

    .. grid-item-card:: Tensor Programming
        :img-top: /_images/optimization_tensor_icon.svg
        :link: optimization_philosophy_tensor_programming
        :link-type: ref

        Use N-dimensional arrays and operations to work with your data directly
        and succinctly.

Example Usage
=============

.. include:: README.rst
  :start-after: start_optimization_examples
  :end-before: end_optimization_examples

Usage Information
=================

*   :ref:`index_concepts` for terminology
*   :ref:`opt_model_construction_nl` for an introduction to using this package
    to model problems.
*   :ref:`opt_index_get_started` for an introduction to optimizing with
    :term:`hybrid` :term:`solvers <solver>`.
*   :ref:`opt_solver_nl_properties` and :ref:`opt_solver_nl_parameters` for the
    solver's properties and parameters.
*   :ref:`opt_index_improving_solutions` for best practices and examples.
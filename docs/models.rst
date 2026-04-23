.. _optimization_models:

================
Nonlinear Models
================

This page describes the `dwave-optimization` package's nonlinear model: classes,
attributes, and methods.

For an introduction to formulating problems as nonlinear models, see the
:ref:`opt_model_construction_nl` section.
The :ref:`opt_example_nl_tsp` demonstrates a simple use of the
`Leap <https://cloud.dwavesys.com/leap/>`_ :term:`hybrid` nonlinear
:term:`solver` on a problem formulated as a nonlinear model; for a more-advanced
end-to-end example, see the :ref:`opt_example_nl_cvrp` section.

.. currentmodule:: dwave.optimization

.. automodule:: dwave.optimization.model

Model Class
===========

.. autoclass:: Model
    :members:
    :inherited-members:
    :member-order: bysource

Expressions
===========

.. currentmodule:: dwave.optimization.expression

.. automodule:: dwave.optimization.expression
    :members:
    :member-order: bysource

.. autoclass:: Expression
    :members:
    :inherited-members:
    :member-order: bysource

States Class
============

.. currentmodule:: dwave.optimization.states

.. autoclass:: States
    :members:
    :inherited-members:
    :member-order: bysource

Functions
=========

.. currentmodule:: dwave.optimization.model

.. automethod:: dwave.optimization.model.Model.constant.clear_cache

.. autofunction:: locked
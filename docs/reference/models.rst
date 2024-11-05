.. _optimization_models:

================
Nonlinear Models 
================

This page describes the `dwave-optimization` package's nonlinear model: classes,
attributes, and methods.

For examples, see
`Ocean's Getting Started examples <https://docs.ocean.dwavesys.com/en/stable/getting_started.html#examples>`_.

.. currentmodule:: dwave.optimization

.. automodule:: dwave.optimization.model

Model Class
-----------

.. autoclass:: Model

Model Attributes 
----------------

.. autosummary::
    :toctree: generated/

    ~Model.objective
    ~Model.states

Model Primitives  
----------------

Instantiation of the model's decision (and constant) symbols. 
For the full list of supported symbols, see the :ref:`optimization_symbols`
page.

.. autosummary::
    :toctree: generated/

    ~Model.binary
    ~Model.constant
    ~Model.disjoint_bit_sets
    ~Model.disjoint_lists
    ~Model.integer
    ~Model.list
    ~Model.set

Model Methods
-------------

.. autosummary::
    :toctree: generated/

    ~Model.add_constraint
    ~Model.decision_state_size
    ~Model.from_file
    ~Model.into_file
    ~Model.is_locked
    ~Model.iter_constraints
    ~Model.iter_decisions
    ~Model.iter_symbols
    ~Model.lock
    ~Model.minimize
    ~Model.num_constraints
    ~Model.num_decisions
    ~Model.num_nodes
    ~Model.num_symbols
    ~Model.quadratic_model
    ~Model.remove_unused_symbols
    ~Model.state_size
    ~Model.to_file
    ~Model.to_networkx
    ~Model.unlock
   
States Class
------------

.. currentmodule:: dwave.optimization.model

.. autoclass:: States

States Methods
--------------

.. autosummary::
   :toctree: generated/

   ~States.clear
   ~States.from_file
   ~States.from_future
   ~States.into_file
   ~States.resize
   ~States.resolve
   ~States.size
   ~States.to_file

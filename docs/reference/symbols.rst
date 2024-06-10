.. _optimization_symbols:

=======
Symbols
=======

Symbols are a model's decision variables, intermediate variables, constants, 
and mathematical operations.

.. _symbols_model_symbols:

Model Symbols
=============

Some of the methods for these symbols (e.g., file operations) are 
intended mostly for package developers. 

.. currentmodule:: dwave.optimization.symbols

.. autosummary::
    :toctree: generated/

    ~Absolute
    ~Add
    ~All
    ~And
    ~AdvancedIndexing
    ~BasicIndexing
    ~BinaryVariable
    ~Constant
    ~DisjointBitSets
    ~DisjointBitSet
    ~DisjointLists
    ~DisjointList
    ~Equal
    ~IntegerVariable
    ~LessEqual
    ~ListVariable
    ~Max
    ~Maximum
    ~Min
    ~Minimum
    ~Multiply
    ~NaryAdd
    ~NaryMaximum
    ~NaryMinimum
    ~NaryMultiply
    ~Negative
    ~Or
    ~Permutation
    ~Prod
    ~QuadraticModel
    ~Reshape
    ~Subtract
    ~SetVariable
    ~Square
    ~Sum
     
Inherited Methods
=================
   
.. currentmodule:: dwave.optimization.model

.. autoclass:: ArrayObserver

The following :class:`~dwave.optimization.model.ArrayObserver` methods
are inherited by the :ref:`model symbols <symbols_model_symbols>`.

.. autosummary::
    :toctree: generated/

    ~ArrayObserver.all
    ~ArrayObserver.has_state
    ~ArrayObserver.max
    ~ArrayObserver.maybe_equals
    ~ArrayObserver.min
    ~ArrayObserver.ndim
    ~ArrayObserver.prod
    ~ArrayObserver.reshape
    ~ArrayObserver.sum
    ~ArrayObserver.shape
    ~ArrayObserver.size
    ~ArrayObserver.state
    ~ArrayObserver.state_size
    ~ArrayObserver.strides

.. autoclass:: NodeObserver

The following :class:`~dwave.optimization.model.NodeObserver` methods
are inherited by the :class:`~dwave.optimization.model.ArrayObserver`
class and :ref:`model symbols <symbols_model_symbols>`.

.. autosummary::
    :toctree: generated/

    ~NodeObserver.equals
    ~NodeObserver.has_state
    ~NodeObserver.iter_predecessors
    ~NodeObserver.iter_successors
    ~NodeObserver.maybe_equals
    ~NodeObserver.reset_state
    ~NodeObserver.shares_memory
    ~NodeObserver.state_size
    ~NodeObserver.topological_index

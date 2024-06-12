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

.. autoclass:: ArraySymbol

The following :class:`~dwave.optimization.model.ArraySymbol` methods
are inherited by the :ref:`model symbols <symbols_model_symbols>`.

.. autosummary::
    :toctree: generated/

    ~ArraySymbol.all
    ~ArraySymbol.has_state
    ~ArraySymbol.max
    ~ArraySymbol.maybe_equals
    ~ArraySymbol.min
    ~ArraySymbol.ndim
    ~ArraySymbol.prod
    ~ArraySymbol.reshape
    ~ArraySymbol.sum
    ~ArraySymbol.shape
    ~ArraySymbol.size
    ~ArraySymbol.state
    ~ArraySymbol.state_size
    ~ArraySymbol.strides

.. autoclass:: Symbol

The following :class:`~dwave.optimization.model.Symbol` methods
are inherited by the :class:`~dwave.optimization.model.ArraySymbol`
class and :ref:`model symbols <symbols_model_symbols>`.

.. autosummary::
    :toctree: generated/

    ~Symbol.equals
    ~Symbol.has_state
    ~Symbol.iter_predecessors
    ~Symbol.iter_successors
    ~Symbol.maybe_equals
    ~Symbol.reset_state
    ~Symbol.shares_memory
    ~Symbol.state_size
    ~Symbol.topological_index
